function training = npd3qn_training(scene, cfg)
numAgents = size(cfg.starts, 1);
agents = cell(numAgents, 1);
for i = 1:numAgents
    agents{i} = npd3qn_net('init', cfg);
end
replays = cell(numAgents, 1);
nBuffers = cell(numAgents, 1);
for i = 1:numAgents
    replays{i} = init_replay(cfg);
    nBuffers{i} = init_n_buffer(cfg.nStep);
end
mission = struct('starts', cfg.starts, 'goals', cfg.goals);
cfg = initialize_reward_norm(cfg);
bestAgents = agents;
bestScore = -inf;
epsilon = cfg.epsilonStart;
globalStep = 0;
episodeReward = zeros(cfg.maxEpisodes, 1);
rawEpisodeReward = zeros(cfg.maxEpisodes, 1);
normalizedStepEpisodeReward = zeros(cfg.maxEpisodes, 1);
successRate = zeros(cfg.maxEpisodes, 1);
lossTrace = nan(cfg.maxEpisodes, numAgents);
positiveRatioTrace = nan(cfg.maxEpisodes, numAgents);
rhoTrace = nan(cfg.maxEpisodes, numAgents);
etaTrace = nan(cfg.maxEpisodes, numAgents);
rewardNormMinTrace = nan(cfg.maxEpisodes, 1);
rewardNormMaxTrace = nan(cfg.maxEpisodes, 1);
for ep = 1:cfg.maxEpisodes
    [envState, observations] = npd3qn_env('reset', scene, cfg, mission);
    agentEpisodeReward = zeros(numAgents, 1);
    agentRawEpisodeReward = zeros(numAgents, 1);
    agentLoss = nan(numAgents, 1);
    while any(envState.active) && envState.step < cfg.maxStepsPerEpisode
        globalStep = globalStep + 1;
        actions = ones(numAgents, 1);
        for i = 1:numAgents
            if ~envState.active(i)
                continue
            end
            if rand < epsilon
                actions(i) = randi(cfg.numActions);
            else
                q = npd3qn_net('predict', agents{i}.online, observations(:, i), cfg.useGPU);
                [~, actions(i)] = max(q);
            end
        end
        [nextEnv, nextObs, envRewards, dones, info] = npd3qn_env('step', scene, cfg, envState, actions);
        rawRewards = info.rawRewards;
        rewards = normalize_step_rewards_from_cumulative_range(rawRewards, cfg);
        for i = 1:numAgents
            if ~envState.active(i)
                continue
            end
            hasCollision = info.collisionObs(i) || info.collisionUav(i);
            isCloser = info.distanceCurr(i) < info.distancePrev(i) - cfg.per.progressTol;
            isPositive = info.reachGoal(i) || ((~hasCollision) && isCloser);
            transition = struct();
            transition.state = observations(:, i);
            transition.action = actions(i);
            transition.reward = rewards(i);
            transition.nextState = nextObs(:, i);
            transition.done = dones(i);
            transition.isPositive = isPositive;
            nBuffers{i} = push_n_buffer(nBuffers{i}, transition);
            [replays{i}, nBuffers{i}] = emit_n_step(replays{i}, nBuffers{i}, cfg);
            agentEpisodeReward(i) = agentEpisodeReward(i) + rewards(i);
            agentRawEpisodeReward(i) = agentRawEpisodeReward(i) + rawRewards(i);
        end
        envState = nextEnv;
        observations = nextObs;
        epsilon = linear_epsilon(cfg, globalStep);
        if mod(globalStep, cfg.updateEverySteps) == 0
            beta = annealed_beta(cfg, globalStep);
            for i = 1:numAgents
                if replays{i}.count < cfg.trainStart
                    continue
                end
                [batch, idxs] = sample_replay(replays{i}, cfg, beta);
                [agents{i}, agentLoss(i), tdAbs] = npd3qn_net('train', agents{i}, batch, cfg, globalStep, cfg.useGPU);
                replays{i} = update_priorities(replays{i}, idxs, tdAbs, cfg);
            end
        end
        if mod(globalStep, cfg.targetUpdateFrequency) == 0
            for i = 1:numAgents
                agents{i}.target = npd3qn_net('copy', agents{i}.target, agents{i}.online);
            end
        end
    end
    for i = 1:numAgents
        [replays{i}, nBuffers{i}] = flush_n_buffer(replays{i}, nBuffers{i}, cfg);
        positiveRatioTrace(ep, i) = replay_positive_ratio(replays{i});
        rhoTrace(ep, i) = cfg.rho;
        etaTrace(ep, i) = cfg.eta;
    end
    normalizedStepEpisodeReward(ep) = sum(agentEpisodeReward);
    rawEpisodeReward(ep) = sum(agentRawEpisodeReward);
    [cfg, rewardNormMinTrace(ep), rewardNormMaxTrace(ep)] = update_reward_norm_from_episode(cfg, rawEpisodeReward(ep));
    episodeReward(ep) = normalize_cumulative_return(rawEpisodeReward(ep), cfg);
    successRate(ep) = mean(envState.success);
    lossTrace(ep, :) = agentLoss;
    if ep >= 50
        currentScore = mean(episodeReward(ep-49:ep));
    else
        currentScore = mean(episodeReward(1:ep));
    end
    if currentScore > bestScore
        bestScore = currentScore;
        bestAgents = agents;
    end
    if mod(ep, 10) == 0 || ep == 1
        fprintf('Episode %d | Reward %.4f | Success %.4f | Epsilon %.4f | Rpos %.4f | rho %.4f | eta %.4f\n', ...
            ep, episodeReward(ep), successRate(ep), epsilon, ...
            mean(positiveRatioTrace(ep, :), 'omitnan'), mean(rhoTrace(ep, :), 'omitnan'), mean(etaTrace(ep, :), 'omitnan'));
    end
end
training = struct();
training.bestAgents = bestAgents;
training.finalAgents = agents;
training.episodeReward = episodeReward;
training.rawEpisodeReward = rawEpisodeReward;
training.normalizedStepEpisodeReward = normalizedStepEpisodeReward;
training.successRate = successRate;
training.lossTrace = lossTrace;
training.positiveRatioTrace = positiveRatioTrace;
training.rhoTrace = rhoTrace;
training.etaTrace = etaTrace;
training.rewardNormMinTrace = rewardNormMinTrace;
training.rewardNormMaxTrace = rewardNormMaxTrace;
training.rewardNorm = cfg.rewardNorm;
training.bestScore = bestScore;
training.finalEpsilon = epsilon;
training.totalSteps = globalStep;
end
function replay = init_replay(cfg)
cap = round(cfg.replayCapacity);
replay = struct();
replay.count = 0;
replay.cursor = 1;
replay.states = cell(cap, 1);
replay.actions = zeros(cap, 1, 'uint16');
replay.returns = zeros(cap, 1, 'single');
replay.nextStates = cell(cap, 1);
replay.dones = false(cap, 1);
replay.gammaPows = zeros(cap, 1, 'single');
replay.isPositive = false(cap, 1);
replay.priorities = ones(cap, 1, 'single') * cfg.per.priorityEps;
replay.maxPriority = single(1);
end
function nBuffer = init_n_buffer(nStep)
nBuffer = struct();
nBuffer.capacity = nStep;
nBuffer.len = 0;
nBuffer.state = cell(nStep, 1);
nBuffer.action = zeros(nStep, 1, 'uint16');
nBuffer.reward = zeros(nStep, 1, 'single');
nBuffer.nextState = cell(nStep, 1);
nBuffer.done = false(nStep, 1);
nBuffer.isPositive = false(nStep, 1);
end
function nBuffer = push_n_buffer(nBuffer, tr)
nBuffer.len = nBuffer.len + 1;
nBuffer.state{nBuffer.len} = tr.state;
nBuffer.action(nBuffer.len) = uint16(tr.action);
nBuffer.reward(nBuffer.len) = single(tr.reward);
nBuffer.nextState{nBuffer.len} = tr.nextState;
nBuffer.done(nBuffer.len) = tr.done;
nBuffer.isPositive(nBuffer.len) = tr.isPositive;
end
function [replay, nBuffer] = emit_n_step(replay, nBuffer, cfg)
while nBuffer.len >= cfg.nStep
    [transition, nBuffer] = pop_n_transition(nBuffer, cfg);
    if accept_replay_transition(transition, replay, cfg)
        replay = add_transition(replay, transition);
    end
end
if nBuffer.len > 0 && nBuffer.done(nBuffer.len)
    [replay, nBuffer] = flush_n_buffer(replay, nBuffer, cfg);
end
end
function [replay, nBuffer] = flush_n_buffer(replay, nBuffer, cfg)
while nBuffer.len > 0
    [transition, nBuffer] = pop_n_transition(nBuffer, cfg);
    if accept_replay_transition(transition, replay, cfg)
        replay = add_transition(replay, transition);
    end
end
end
function [transition, nBuffer] = pop_n_transition(nBuffer, cfg)
steps = min(cfg.nStep, nBuffer.len);
ret = single(0);
g = single(1);
done = false;
positive = nBuffer.isPositive(1);
for k = 1:steps
    ret = ret + g * single(nBuffer.reward(k));
    if nBuffer.done(k)
        done = true;
        steps = k;
        break
    end
    g = g * single(cfg.gamma);
end
transition = struct();
transition.state = nBuffer.state{1};
transition.action = double(nBuffer.action(1));
transition.returns = ret;
transition.nextState = nBuffer.nextState{steps};
transition.done = done;
transition.gammaPows = single(cfg.gamma ^ steps);
transition.isPositive = positive;
for k = 2:nBuffer.len
    nBuffer.state{k-1} = nBuffer.state{k};
    nBuffer.action(k-1) = nBuffer.action(k);
    nBuffer.reward(k-1) = nBuffer.reward(k);
    nBuffer.nextState{k-1} = nBuffer.nextState{k};
    nBuffer.done(k-1) = nBuffer.done(k);
    nBuffer.isPositive(k-1) = nBuffer.isPositive(k);
end
nBuffer.state{nBuffer.len} = [];
nBuffer.nextState{nBuffer.len} = [];
nBuffer.action(nBuffer.len) = uint16(0);
nBuffer.reward(nBuffer.len) = single(0);
nBuffer.done(nBuffer.len) = false;
nBuffer.isPositive(nBuffer.len) = false;
nBuffer.len = nBuffer.len - 1;
end
function replay = add_transition(replay, tr)
idx = replay.cursor;
replay.states{idx} = tr.state;
replay.actions(idx) = uint16(tr.action);
replay.returns(idx) = single(tr.returns);
replay.nextStates{idx} = tr.nextState;
replay.dones(idx) = tr.done;
replay.gammaPows(idx) = single(tr.gammaPows);
replay.isPositive(idx) = tr.isPositive;
replay.priorities(idx) = replay.maxPriority;
replay.count = min(replay.count + 1, numel(replay.actions));
replay.cursor = idx + 1;
if replay.cursor > numel(replay.actions)
    replay.cursor = 1;
end
end
function [batch, idxs] = sample_replay(replay, cfg, beta)
N = replay.count;
priority = double(replay.priorities(1:N));
Rpos = replay_positive_ratio(replay);
positiveMask = replay.isPositive(1:N);
priorityStar = priority;
priorityStar(positiveMask) = cfg.eta .* priorityStar(positiveMask);
mass = priorityStar .^ cfg.per.alpha;
sumMass = sum(mass);
if ~(isfinite(sumMass) && sumMass > 0)
    P = ones(N, 1) / N;
else
    P = mass / sumMass;
end
cdf = cumsum(P);
idxs = zeros(cfg.batchSize, 1);
for i = 1:cfg.batchSize
    r = rand;
    idxs(i) = find(cdf >= r, 1, 'first');
    if isempty(idxs(i))
        idxs(i) = N;
    end
end
Ptilde = max(P(idxs), cfg.per.probFloor);
wHat = (N .* Ptilde) .^ (-beta);
wClip = min(max(wHat, cfg.per.weightMin), cfg.per.weightMax);
weights = wClip ./ max(wClip);
batch = struct();
batch.states = replay.states(idxs);
batch.actions = double(replay.actions(idxs));
batch.returns = double(replay.returns(idxs));
batch.nextStates = replay.nextStates(idxs);
batch.dones = double(replay.dones(idxs));
batch.gammaPows = double(replay.gammaPows(idxs));
batch.isWeights = double(weights(:)');
batch.positiveRatio = Rpos;
batch.eta = cfg.eta;
end
function accept = accept_replay_transition(transition, ~, cfg)
if transition.isPositive
    accept = true;
else
    accept = rand < cfg.rho;
end
end
function Rpos = replay_positive_ratio(replay)
N = replay.count;
if N > 0
    Rpos = mean(double(replay.isPositive(1:N)));
else
    Rpos = NaN;
end
end
function replay = update_priorities(replay, idxs, tdAbs, cfg)
p = single(abs(tdAbs(:)) + cfg.per.priorityEps);
for k = 1:numel(idxs)
    replay.priorities(idxs(k)) = p(k);
end
replay.maxPriority = max(replay.maxPriority, max(p));
end
function epsilon = linear_epsilon(cfg, globalStep)
ratio = min(1, globalStep / cfg.epsilonDecaySteps);
epsilon = cfg.epsilonStart + ratio * (cfg.epsilonEnd - cfg.epsilonStart);
end
function rewards = normalize_step_rewards_from_cumulative_range(rawRewards, cfg)
if ~isfield(cfg, 'rewardNorm') || ~isfield(cfg.rewardNorm, 'enabled') || ~cfg.rewardNorm.enabled || ~isfield(cfg.rewardNorm, 'normalizeStepRewards') || ~cfg.rewardNorm.normalizeStepRewards
    rewards = rawRewards;
    return
end
rMin = cfg.rewardNorm.initialMin;
rMax = cfg.rewardNorm.initialMax;
if isempty(rMin) || isempty(rMax) || ~isfinite(rMin) || ~isfinite(rMax) || abs(rMax - rMin) < cfg.rewardNorm.eps
    rewards = rawRewards;
    return
end
rewards = (rawRewards - rMin) ./ (rMax - rMin);
if isfield(cfg.rewardNorm, 'clip') && cfg.rewardNorm.clip
    rewards = min(max(rewards, 0), 1);
end
end
function rNorm = normalize_cumulative_return(episodeReturn, cfg)
if ~isfield(cfg, 'rewardNorm') || ~isfield(cfg.rewardNorm, 'enabled') || ~cfg.rewardNorm.enabled
    rNorm = episodeReturn;
    return
end
rMin = cfg.rewardNorm.observedMin;
rMax = cfg.rewardNorm.observedMax;
if isempty(rMin) || isempty(rMax) || ~isfinite(rMin) || ~isfinite(rMax) || abs(rMax - rMin) < cfg.rewardNorm.eps
    rNorm = episodeReturn;
    return
end
rNorm = (episodeReturn - rMin) ./ (rMax - rMin);
if isfield(cfg.rewardNorm, 'clip') && cfg.rewardNorm.clip
    rNorm = min(max(rNorm, 0), 1);
end
end
function cfg = initialize_reward_norm(cfg)
if ~isfield(cfg, 'rewardNorm') || ~isfield(cfg.rewardNorm, 'enabled') || ~cfg.rewardNorm.enabled
    return
end
if ~isfield(cfg.rewardNorm, 'observedMin') || isempty(cfg.rewardNorm.observedMin)
    cfg.rewardNorm.observedMin = inf;
end
if ~isfield(cfg.rewardNorm, 'observedMax') || isempty(cfg.rewardNorm.observedMax)
    cfg.rewardNorm.observedMax = -inf;
end
if ~isfield(cfg.rewardNorm, 'initialMin')
    cfg.rewardNorm.initialMin = [];
end
if ~isfield(cfg.rewardNorm, 'initialMax')
    cfg.rewardNorm.initialMax = [];
end
end
function [cfg, rMin, rMax] = update_reward_norm_from_episode(cfg, episodeReturn)
if ~isfield(cfg, 'rewardNorm') || ~isfield(cfg.rewardNorm, 'enabled') || ~cfg.rewardNorm.enabled
    rMin = NaN;
    rMax = NaN;
    return
end
if isfinite(episodeReturn)
    cfg.rewardNorm.observedMin = min(cfg.rewardNorm.observedMin, episodeReturn);
    cfg.rewardNorm.observedMax = max(cfg.rewardNorm.observedMax, episodeReturn);
    cfg.rewardNorm.initialMin = cfg.rewardNorm.observedMin;
    cfg.rewardNorm.initialMax = cfg.rewardNorm.observedMax;
end
rMin = cfg.rewardNorm.observedMin;
rMax = cfg.rewardNorm.observedMax;
end
function beta = annealed_beta(cfg, globalStep)
ratio = min(1, globalStep / max(1, cfg.totalTrainingSteps));
beta = cfg.per.beta0 + ratio * (cfg.per.beta1 - cfg.per.beta0);
end
