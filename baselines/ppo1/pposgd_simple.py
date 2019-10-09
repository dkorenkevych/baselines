from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from scipy.stats import beta
def traj_segment_generator(pi, env, horizon, stochastic, lstm=False):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)]).astype('float32')
    if lstm:
        states_v = np.zeros((horizon, 128), 'float32')
        states_p = np.zeros((horizon, 128), 'float32')
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    state_v = np.zeros((1, 128))
    state_p = np.zeros((1, 128))
    ac_act = ac.copy()
    while True:
        prevac = ac
        old_ac = ac.copy()
        ac, mean, vpred, state_v, state_p = pi.act(stochastic, ob[None], state_v, state_p, np.expand_dims(new, axis=0))
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "states_v": states_v, "states_p": states_p}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        pi.lstm_v.reset_states(np.split(state_v, indices_or_sections=2, axis=-1))
        pi.lstm_p.reset_states(np.split(state_p, indices_or_sections=2, axis=-1))
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac
        states_v[i] = state_v
        states_p[i] = state_p

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
            state_p.fill(0)
            state_v.fill(0)
            pi.lstm_v.reset_states()
            pi.lstm_p.reset_states()
        t += 1




def add_vtarg_and_adv(seg, gamma, lam, value_rms):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"]) #* value_rms.std.eval() + value_rms.mean.eval()
    #vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]
    #value_rms.update(seg["tdlamret"])

def learn(env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        lstm=False):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    #with tf.variable_scope("model", reuse=False):
    #pi_act = policy_fn("pi", ob_space, ac_space, n_steps=1, reuse=False) # Construct network for new policy
    #with tf.variable_scope("model", reuse=True):
    pi = policy_fn("pi", ob_space, ac_space, n_steps=None, reuse=True)
    #with tf.variable_scope("old_model", reuse=False):
    oldpi = policy_fn("oldpi", ob_space, ac_space, n_steps=None, reuse=False) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = pi.ob
    old_ob = oldpi.ob
    #ob = tf.placeholder(tf.float32, (64, 6), name="obs")
    ac = pi.pdtype.sample_placeholder([None])
    states_ph_v = tf.placeholder(tf.float32, (None, 128), name="states_ph_v")
    states_ph_p = tf.placeholder(tf.float32, (None, 128), name="states_ph_p")
    dones_ph = tf.placeholder(tf.float32, (None,), name="dones_ph")

    value_rms = RunningMeanStd(shape=(1,))
    #norm_ret = (ret - value_rms.mean) / value_rms.std
    norm_ret = ret

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - norm_ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    grad_ph = np.zeros((4, sum([np.product(var.shape) for var in var_list])), dtype='float32')
    lossandgrad = U.function([ob, old_ob, ac, atarg, ret, lrmult, pi._states_ph_v, oldpi._states_ph_v, pi._states_ph_p, oldpi._states_ph_p, pi._dones_ph, oldpi._dones_ph], losses + [U.flatgrad(total_loss, var_list, clip_norm=5)])
    grad_counter  =0
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, old_ob, ac, atarg, ret, lrmult, pi._states_ph_v, oldpi._states_ph_v, pi._states_ph_p, oldpi._states_ph_p, pi._dones_ph, oldpi._dones_ph], losses)

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True, lstm=lstm)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam, value_rms)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret, state_v, state_p, new = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["states_v"], seg["states_p"],  seg["new"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret, state_v=state_v, state_p=state_p, new=new), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                pi.lstm_v.reset_states(np.split(batch["state_v"][:1], indices_or_sections=2, axis=-1))
                pi.lstm_p.reset_states(np.split(batch["state_p"][:1], indices_or_sections=2, axis=-1))
                oldpi.lstm_v.reset_states(np.split(batch["state_v"][:1], indices_or_sections=2, axis=-1))
                oldpi.lstm_p.reset_states(np.split(batch["state_p"][:1], indices_or_sections=2, axis=-1))
                *newlosses, g = lossandgrad(batch["ob"][None], batch["ob"][None], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, batch["state_v"], batch["state_v"], batch["state_p"], batch["state_p"], batch["new"], batch["new"])
                # np.copyto(grad_ph[grad_counter % 4], g)
                # grad_counter += 1
                # if grad_counter % 4 == 0:
                #     adam.update(np.mean(grad_ph, axis=0, dtype='float32'), optim_stepsize * cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"][None], batch["ob"][None], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, batch["state_v"], batch["state_v"], batch["state_p"], batch["state_p"], batch["new"], batch["new"])
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
