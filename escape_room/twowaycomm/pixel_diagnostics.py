"""Signaling, causal listening, and frozen CNN probes for pixel checkpoints.

Use physical GPU 2 only. The default set is the six completed arrow runs.
Continuous messages/actions require an adaptation of Lowe et al. (2019):
SC uses binned messages and analytically integrated clipped-Gaussian controls;
CIC is conditional MI under controlled visual/task interventions with a uniform
private-condition prior. Observations/history stay fixed for the action/reply
intervention. This is local causal influence, not a full alternative rollout.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from tensordict import TensorDict

from escape_room.twowaycomm.communication_metrics import (
    action_bin_probabilities, codes_for, information_with_null, listening_summary,
    signaling_step, soft_mutual_information, self_test,
)
from escape_room.twowaycomm.env import make_env
from escape_room.twowaycomm.env_cfg import twowaycomm_ppo_runner_cfg
from escape_room.twowaycomm.evaluate import balanced_conditions, checkpoint_actor_architecture
from escape_room.twowaycomm.feature_probe import probe_actor
from escape_room.twowaycomm.model import TwoWayDialActor

ROOT = Path('/workspaces/communication')
DEFAULT_CHECKPOINTS = [
    ROOT / f'ckpts/twoway_comm/pixel_gru_20261005/mappo_pixel{size}_{layout}_s42/model_final.pt'
    for size in (32, 64) for layout in ('front', 'scattered')
] + [ROOT / f'ckpts/twoway_comm/pixel_gru_192_retry_20261005/mappo_pixel192_{layout}_s42/model_final.pt'
     for layout in ('front', 'scattered')]


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def td(obs, worlds):
    return obs if isinstance(obs, TensorDict) else TensorDict(obs, batch_size=[worlds])


def load_actor(path, env):
    architecture = checkpoint_actor_architecture(path)
    cfg = twowaycomm_ppo_runner_cfg(obs_mode='pixel', algorithm='mappo',
        sender_message_dim=architecture['sender_message_dim'],
        receiver_message_dim=architecture['receiver_message_dim'],
        channel_mode=architecture['channel_mode'],
        delayed_message_feedback=architecture['delayed_message_feedback'],
        message_unit=architecture['message_unit'])
    for key in ('hidden_dims', 'sender_hidden_dims', 'sender_message_hidden_dims',
                'receiver_message_hidden_dims', 'cnn_cfg'):
        if architecture.get(key) is not None:
            setattr(cfg.actor, key, architecture[key])
    obs, _ = env.reset()
    obs = td(obs, env.num_envs)
    kwargs = {k: v for k, v in asdict(cfg.actor).items() if v is not None and
              k not in ('class_name', 'rnn_type', 'rnn_hidden_dim', 'rnn_num_layers')}
    actor = TwoWayDialActor(obs, {'actor': cfg.obs_groups['actor']}, 'actor', 6, **kwargs).to(env.device)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
    actor.eval()
    del checkpoint
    return actor, architecture


def capture(env, actor, conditions):
    obs, _ = env.reset()
    term = env.action_manager.get_term('agents')
    term.set_arrow_directions(conditions.directions)
    term.set_door_color(conditions.colors)
    env.sim.forward()
    env.sim.sense()  # Manually changed task labels must reach the initial pixels.
    obs = td(env.observation_manager.compute(update_history=True), env.num_envs)
    actor.reset()
    previous = (torch.zeros(env.num_envs, actor.sender_message_dim, device=env.device),
                torch.zeros(env.num_envs, actor.receiver_message_dim, device=env.device))
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    correct, wrong, timeout = (torch.zeros_like(active) for _ in range(3))
    frames = []
    for step in range(env.max_episode_length):
        assert not obs['sender'][:, 7:13].any() and not obs['receiver'][:, 7:13].any()
        messages = actor.encode_messages(obs, previous)
        means = actor.action_from_messages(obs, *previous)
        frames.append({'obs': obs.clone(), 'messages': tuple(m.clone() for m in messages),
                       'delivered': tuple(m.clone() for m in previous), 'means': means.clone(),
                       'active': active.clone(), 'poses': term._sim_data.qpos[:, term._q].clone(),
                       'velocities': term._sim_data.qvel[:, term._v].clone()})
        obs, _, terminated, truncated, _ = env.step(means)
        obs = td(obs, env.num_envs)
        previous = messages
        done = terminated | truncated
        latch = done & active
        correct[latch], wrong[latch], timeout[latch] = (
            term.correct_entry[latch], term.wrong_entry[latch], term.timeout[latch])
        active &= ~latch
        if not active.any():
            break
        if done.any():
            env.reset(env_ids=done.nonzero(as_tuple=False).squeeze(-1))
            previous = tuple(torch.where(done[:, None], torch.zeros_like(m), m) for m in previous)
            obs = td(env.observation_manager.compute(update_history=True), env.num_envs)
    if active.any():
        raise RuntimeError('Some worlds failed to finish within the horizon')
    return frames, {'episodes': env.num_envs, 'correct': int(correct.sum()),
                    'wrong': int(wrong.sum()), 'timeout': int(timeout.sum())}


def alternative_messages(env, actor, frame, conditions, channel):
    """Vary only current private visuals, retaining observed pose and history."""
    term = env.action_manager.get_term('agents')
    term._sim_data.qpos[:, term._q] = frame['poses']
    term._sim_data.qvel[:, term._v] = frame['velocities']
    term.set_arrow_directions(conditions.directions)
    term.set_door_color(conditions.colors)
    candidates = []
    count = 8 if channel == 0 else 3
    image_key = ('sender_image', 'receiver_image')[channel]
    sensor_name = ('sender_camera', 'receiver_camera')[channel]
    observed_condition = conditions.condition_id % 8 if channel == 0 else conditions.colors
    for candidate in range(count):
        if channel == 0:
            directions = torch.tensor([1 if candidate & (1 << i) else -1 for i in range(3)], device=env.device)
            term.set_arrow_directions(directions)
        else:
            term.set_door_color(torch.tensor(candidate, device=env.device))
        env.sim.forward()
        env.sim.sense()
        changed = frame['obs'].clone()
        changed[image_key] = env.scene[sensor_name].data.rgb.permute(0, 3, 1, 2).float() / 255
        messages = actor.encode_messages(changed, frame['delivered'])[channel]
        same = (observed_condition == candidate) & frame['active']
        if same.any() and not torch.allclose(messages[same], frame['messages'][channel][same], atol=1e-5, rtol=1e-5):
            raise AssertionError('Unchanged-private-condition message control failed')
        candidates.append(messages.clone())
    return torch.stack(candidates)


def measure_communication(env, actor, frames, conditions, steps, samples, seed):
    if not actor.is_recurrent or not actor.delayed_message_feedback:
        raise ValueError('This diagnostic currently targets delayed-feedback pixel models')
    actor.distribution.update(frames[0]['means'])
    std = actor.output_std[0].clone()
    report = {'reference': 'https://arxiv.org/abs/1903.05168',
              'units': 'bits; Gaussian KL diagnostic uses nats',
              'population': 'census of all 24 balanced task conditions, deterministic trajectories',
              'message_bins_per_coordinate': [4, 8, 16], 'action_bins_per_control': 3,
              'listening_method': 'uniform private-visual interventions, fixed listener observation/history',
              'late_steps_note': 'late measurements condition on surviving episodes; n_active is reported',
              'signaling': {}, 'speaker_consistency': {}, 'listening': {}, 'message_statistics': {}}
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=env.device).manual_seed(seed)
    heads = (actor.sender_head, actor.receiver_head)
    latents = (actor.sender_latent_encoder, actor.receiver_latent_encoder)
    encoders = (actor.sender_message_encoder, actor.receiver_message_encoder)
    for step in steps:
        if step + 1 >= len(frames):
            continue
        frame, future = frames[step], frames[step + 1]
        mask = frame['active'] & future['active']
        ids = mask.nonzero(as_tuple=False).squeeze(-1)
        if not len(ids):
            continue
        subset = type(conditions)(conditions.directions[ids], conditions.colors[ids])
        messages = [m[ids].cpu().numpy() for m in frame['messages']]
        key = str(step)
        report['signaling'][key] = {'n_active': len(ids), 'bins': signaling_step(*messages, subset, rng, False)}
        report['speaker_consistency'][key] = {'n_active': len(ids)}
        report['message_statistics'][key] = {}
        for channel, name in enumerate(('sender', 'receiver')):
            values = frame['messages'][channel][ids]
            report['message_statistics'][key][name] = {
                'coordinate_std': values.std(0, unbiased=False).cpu().tolist(),
                'fraction_abs_gt_0_999': float((values.abs() > .999).float().mean()),
                'mean_tanh_derivative': float((1 - values.square()).mean())}
            codes = codes_for(messages[channel], 8)
            probs = action_bin_probabilities(future['means'][ids, channel * 3:(channel + 1) * 3],
                                             std[channel * 3:(channel + 1) * 3])
            value = soft_mutual_information(codes, probs)
            null = np.mean([soft_mutual_information(rng.permutation(codes), probs) for _ in range(64)])
            report['speaker_consistency'][key][name] = {
                'bits': value, 'permutation_null_mean_bits': float(null),
                'null_corrected_bits': float(value - null)}
        next_features = actor._features(future['obs'][ids])
        report['listening'][key] = {'n_active': len(ids)}
        for channel, name in enumerate(('sender_to_receiver', 'receiver_to_sender')):
            listener = 1 - channel
            incoming = alternative_messages(env, actor, frame, conditions, channel)[:, ids]
            listener_latent = latents[listener](next_features[listener])[None].expand(len(incoming), -1, -1)
            counter = heads[listener](torch.cat((listener_latent, incoming), -1))
            baseline = future['means'][ids, listener * 3:(listener + 1) * 3]
            result = listening_summary(counter, baseline, std[listener * 3:(listener + 1) * 3], samples, generator)
            result['delivery_delay_steps'] = 1
            if step + 2 < len(frames):
                mask2 = mask & frames[step + 2]['active']
                local_ids = mask2[ids].nonzero(as_tuple=False).squeeze(-1)
                ids2 = ids[local_ids]
                if len(ids2):
                    own_features = next_features[listener][local_ids][None].expand(len(incoming), -1, -1)
                    replies = actor._bound(encoders[listener](torch.cat((own_features, incoming[:, local_ids]), -1)))
                    after = frames[step + 2]
                    source_features = actor._features(after['obs'][ids2])[channel]
                    source_latent = latents[channel](source_features)[None].expand(len(incoming), -1, -1)
                    mediated_means = heads[channel](torch.cat((source_latent, replies), -1))
                    mediated = listening_summary(mediated_means,
                        after['means'][ids2, channel * 3:(channel + 1) * 3],
                        std[channel * 3:(channel + 1) * 3], samples, generator)
                    mediated['n_active'] = len(ids2)
                    mediated['outgoing_reply_rms_change'] = float(
                        (replies - future['messages'][listener][ids2][None]).square().mean().sqrt())
                    result['feedback_reply_to_original_speaker_action_t_plus_2'] = mediated
            report['listening'][key][name] = result
        print(f'COMM_PROGRESS step={step} active={len(ids)}', flush=True)
    return report


def summary_row(record):
    report = record['communication']
    key = '1' if '1' in report['signaling'] else next(iter(report['signaling']))
    signaling = report['signaling'][key]['bins']['8']
    listening = report['listening'][key]
    return {'run': record['run'], 'pixel_size': record['environment']['pixel_size'],
            'layout': record['environment']['arrow_layout'], 'headline_step': int(key),
            'sender_target_mi_bits': signaling['sender_target_given_colour']['bits'],
            'receiver_color_mi_bits': signaling['receiver_door_colour']['bits'],
            'sender_sc_bits': report['speaker_consistency'][key]['sender']['bits'],
            'receiver_sc_bits': report['speaker_consistency'][key]['receiver']['bits'],
            's_to_r_cic_bits': listening['sender_to_receiver']['cic_bits'],
            'r_to_s_cic_bits': listening['receiver_to_sender']['cic_bits'],
            'r_to_s_reply_to_r_cic_bits': listening['receiver_to_sender'].get(
                'feedback_reply_to_original_speaker_action_t_plus_2', {}).get('cic_bits'),
            'sender_feature_direction_spawn_accuracy': float(np.mean([
                record['feature_probes']['datasets'][f'spawn/sender/direction_{i}']['trained_cnn']['test_accuracy']
                for i in range(3)])),
            'sender_feature_direction_near_accuracy': float(np.mean([
                record['feature_probes']['datasets'][f'near/sender/direction_{i}']['trained_cnn']['test_accuracy']
                for i in range(3)])),
            'receiver_feature_color_spawn_accuracy': record['feature_probes']['datasets'][
                'spawn/receiver/door_color']['trained_cnn']['test_accuracy']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', nargs='+', type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'ckpts/twoway_comm/pixel_diagnostics_20261007')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--mc-samples', type=int, default=256)
    parser.add_argument('--probe-poses', nargs=3, type=int, default=[8, 4, 8])
    parser.add_argument('--steps', nargs='+', type=int, default=[0, 1, 2, 4, 8, 16, 32, 64])
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.self_test:
        self_test()
        return
    if args.device.startswith('cuda') and os.environ.get('CUDA_VISIBLE_DEVICES') != '2':
        raise SystemExit('This experiment is authorized only on physical GPU 2')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, path in enumerate(args.checkpoints, 1):
        started = time.monotonic()
        architecture = checkpoint_actor_architecture(path)
        environment = dict(architecture['environment'])
        env = make_env(num_envs=24, device=args.device, seed=args.seed, auto_reset=False, **environment)
        try:
            actor, _ = load_actor(path, env)
            versions = [p._version for p in actor.parameters()]
            conditions = balanced_conditions(24, env.device, args.seed)
            with torch.inference_mode():
                frames, performance = capture(env, actor, conditions)
                communication = measure_communication(env, actor, frames, conditions, args.steps,
                                                       args.mc_samples, args.seed)
                del frames
                feature_probes = probe_actor(env, actor, conditions, args.probe_poses, args.seed)
            if versions != [p._version for p in actor.parameters()]:
                raise AssertionError('The frozen actor parameters were modified')
            record = {'run': path.parent.name, 'checkpoint': str(path.resolve()),
                      'environment': environment, 'measurement_seed': args.seed,
                      'measured_at_utc': datetime.now(timezone.utc).isoformat(),
                      'fresh_camera_render_after_manual_task_reset': True,
                      'actor_parameters_unchanged': True, 'census_performance': performance,
                      'communication': communication, 'feature_probes': feature_probes}
            record['elapsed_seconds'] = time.monotonic() - started
            write_json(args.output_dir / (record['run'] + '.diagnostics.json'), record)
            records.append(record)
            write_json(args.output_dir / 'results.json', {'runs': records})
            rows = [summary_row(r) for r in records]
            with (args.output_dir / 'summary.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print('DIAGNOSTICS_RUN_COMPLETED ' + json.dumps(rows[-1]), flush=True)
            print(f'DIAGNOSTICS_PROGRESS {index}/{len(args.checkpoints)} elapsed={record["elapsed_seconds"]:.1f}s', flush=True)
            del actor
        finally:
            env.close()
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    print('DIAGNOSTICS_COMPLETED ' + str(args.output_dir), flush=True)


if __name__ == '__main__':
    main()
