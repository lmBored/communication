"""Measure signaling and listening in existing TwoWayComm checkpoints.

Reference: Lowe et al., On the Pitfalls of Measuring Emergent Communication,
arXiv:1903.05168, definitions 3.1/3.2, Eq. (2), Algorithm 1.

SC is message/future-own-action MI, using 3 bins per clamped control and
analytically integrated Gaussian action probabilities. Observation/message
MI complements SC because the sender's movements are not the task answer.
Tanh messages use 4/8/16 equal-width bins per coordinate; DRU bits are exact.

CIC is a conditional empirical-codebook adaptation, NOT a replication of the
paper's categorical message policy: evaluation messages here are deterministic.
For each fixed listener observation/history we intervene with actual messages
from each alternative speaker-private task condition at the same timestep.
S->R enumerates 8 arrow patterns at the listener's fixed door colour; R->S
enumerates 3 colours at the listener's fixed arrow pattern. Priors are uniform.
The policy's Gaussian actions are clipped to [-1,1], including boundary atoms.
CIC = mean_k KL(p(clipped_action | message_k) || mean_j p(action | message_j)).
We integrate this Gaussian mixture by Monte Carlo, report bits, and separately
measure the pure message-feedback->reply->next-action path with observations
fixed to the normal trajectory. This is not a full counterfactual trajectory.

Run from /workspaces/communication inside its venv, with MUJOCO_GL=disable and
CUDA_VISIBLE_DEVICES=2 (logical --device cuda:0). No training, repo writes, or
W&B logging occur. Results are printed as RUN_METRICS_JSON records.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import io
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from tensordict import TensorDict


def mi_joint(joint):
    joint=np.asarray(joint,dtype=np.float64)
    joint=joint/joint.sum()
    product=joint.sum(axis=1,keepdims=True)*joint.sum(axis=0,keepdims=True)
    selected=joint>0
    return float(np.sum(joint[selected]*np.log2(joint[selected]/product[selected])))


def mutual_information(x,y):
    _,x=np.unique(np.asarray(x),return_inverse=True)
    _,y=np.unique(np.asarray(y),return_inverse=True)
    joint=np.bincount(x*(y.max()+1)+y,minlength=(x.max()+1)*(y.max()+1)).reshape(x.max()+1,y.max()+1)
    return mi_joint(joint)


def soft_mutual_information(codes,probabilities):
    _,codes=np.unique(codes,return_inverse=True)
    joint=np.zeros((codes.max()+1,probabilities.shape[1]),dtype=np.float64)
    np.add.at(joint,codes,probabilities)
    return mi_joint(joint)


def conditional_information(codes,labels,condition):
    return sum(np.mean(condition==c)*mutual_information(codes[condition==c],labels[condition==c]) for c in np.unique(condition))


def codes_for(messages,bins,discrete=False):
    messages=np.asarray(messages)
    digits=(messages>.5).astype(np.int64) if discrete else np.minimum(bins-1,np.maximum(0,np.floor((messages+1)*bins/2))).astype(np.int64)
    radix=2 if discrete else bins
    return (digits*(radix**np.arange(messages.shape[-1]))).sum(axis=-1)


def information_with_null(codes,labels,rng,condition=None,permutations=16):
    metric=lambda x: mutual_information(x,labels) if condition is None else conditional_information(x,labels,condition)
    observed=metric(codes)
    null=[]
    for _ in range(permutations):
        shuffled=codes.copy()
        if condition is None:
            rng.shuffle(shuffled)
        else:
            for c in np.unique(condition):
                ids=np.flatnonzero(condition==c)
                shuffled[ids]=codes[rng.permutation(ids)]
        null.append(metric(shuffled))
    return {'bits':float(observed),'permutation_null_mean_bits':float(np.mean(null)),'null_corrected_bits':float(observed-np.mean(null))}


def action_bin_probabilities(means,std,bins=3):
    # Clipping folds Gaussian tails into the first and last control bins.
    edges=torch.linspace(-1,1,bins+1,device=means.device,dtype=means.dtype)[1:-1]
    cdf=torch.special.ndtr((edges[None,None,:]-means[:,:,None])/std[None,:,None])
    cdf=torch.cat((torch.zeros_like(cdf[:,:,:1]),cdf,torch.ones_like(cdf[:,:,:1])),dim=-1)
    per_dimension=cdf.diff(dim=-1)
    columns=[per_dimension[:,0,i]*per_dimension[:,1,j]*per_dimension[:,2,k] for i,j,k in itertools.product(range(bins),repeat=3)]
    return torch.stack(columns,dim=-1).cpu().numpy()


def clipped_gaussian_cic(means,std,samples,generator):
    """means: [candidate, listener-state, 3]; output MI per listener-state.

    The density uses a mixed dominating measure: interior Lebesgue density plus
    atomic probabilities at -1/+1. Interior normalizing constants cancel between
    mixture components (same std and the same boundary/interior coordinates).
    """
    candidates,worlds,dimensions=means.shape
    noise=torch.randn((candidates,worlds,samples,dimensions),device=means.device,dtype=means.dtype,generator=generator)
    actions=(means[:,:,None,:]+std[None,None,None,:]*noise).clamp(-1,1)
    distance=(actions[:,None,:,:,:]-means[None,:,:,None,:])/std[None,None,None,None,:]
    logp=-.5*distance.square()
    lower=torch.special.log_ndtr((-1-means)/std)
    upper=torch.special.log_ndtr((means-1)/std)
    logp=torch.where(actions[:,None]<=-1,lower[None,:,:,None,:],logp)
    logp=torch.where(actions[:,None]>=1,upper[None,:,:,None,:],logp).sum(dim=-1)
    own=logp.diagonal(dim1=0,dim2=1).permute(2,0,1)
    mixture=torch.logsumexp(logp,dim=1)-math.log(candidates)
    estimate=((own-mixture)/math.log(2)).mean(dim=(0,2))
    # MC variance is recorded over independent Gaussian draws, conditional on
    # the fixed trajectory states and empirical intervention codebook.
    draw_means=((own-mixture)/math.log(2)).mean(dim=(0,1))
    return {'cic_bits':float(estimate.mean()),'mc_standard_error_bits':float(draw_means.std(unbiased=True)/math.sqrt(samples)),'min_state_bits':float(estimate.min()),'max_state_bits':float(estimate.max()),'candidate_count':candidates,'upper_bound_bits':math.log2(candidates)}


def listening_summary(means,baseline,std,samples,generator):
    summary=clipped_gaussian_cic(means,std,samples,generator)
    summary['clipped_mean_action_rms_change']=float((means.clamp(-1,1)-baseline[None].clamp(-1,1)).square().mean().sqrt())
    summary['preclip_gaussian_kl_to_observed_nats']=float(.5*((means-baseline[None])/std).square().sum(dim=-1).mean())
    # The identical-message control must carry no conditional influence.
    null=clipped_gaussian_cic(baseline[None].expand_as(means),std,min(8,samples),generator)
    summary['identical_message_control_cic_bits']=null['cic_bits']
    if abs(null['cic_bits'])>1e-5:
        raise AssertionError(f'CIC null control failed: {null}')
    return summary


def signaling_step(sender,receiver,conditions,rng,discrete):
    colours=conditions.colors.cpu().numpy()
    arrows=((conditions.directions==1).long()*(2**torch.arange(3,device=conditions.directions.device))).sum(-1).cpu().numpy()
    target=(conditions.target_direction==1).cpu().numpy().astype(int)
    result={}
    for bins in ((2,) if discrete else (4,8,16)):
        sm,rm=codes_for(sender,bins,discrete),codes_for(receiver,bins,discrete)
        result[str(bins)]={'sender_arrow_pattern':information_with_null(sm,arrows,rng),'sender_target_given_colour':information_with_null(sm,target,rng,colours),'receiver_door_colour':information_with_null(rm,colours,rng)}
    return result


def self_test():
    assert abs(mutual_information(np.tile([0,0,1,1],20),np.tile([0,1,0,1],20)))<1e-12
    assert abs(mutual_information(np.tile([0,1],40),np.tile([0,1],40))-1)<1e-12
    x=np.tile([0,0,1,1],20); c=np.tile([0,1,0,1],20)
    assert abs(conditional_information(x^c,x,c)-1)<1e-12
    assert codes_for(np.array([[-1,-1],[1,1]]),8).tolist()==[0,63]
    assert codes_for(np.array([[0,1],[1,0]]),2,True).tolist()==[2,1]
    generator=torch.Generator().manual_seed(11)
    means=torch.zeros(3,20,3)
    null=clipped_gaussian_cic(means,torch.ones(3),128,generator)
    assert abs(null['cic_bits'])<1e-6
    saturated=torch.tensor([3.,5.])[:,None,None].expand(2,20,3)
    assert abs(clipped_gaussian_cic(saturated,torch.full((3,),.1),128,generator)['cic_bits'])<1e-6
    separated=torch.tensor([-3.,3.])[:,None,None].expand(2,20,3)
    assert abs(clipped_gaussian_cic(separated,torch.full((3,),.1),128,generator)['cic_bits']-1)<1e-6
    means=torch.tensor([[-.5,0,0],[.5,0,0]]).repeat_interleave(40,dim=0)
    probabilities=action_bin_probabilities(means,torch.full((3,),.1))
    assert np.allclose(probabilities.sum(-1),1)
    assert soft_mutual_information(np.repeat([0,1],40),probabilities)>.8
    print('METRICS_SELF_TEST PASS (MI, conditional MI, message bins, Gaussian bins, CIC null, clipping, 1-bit influence)',flush=True)

