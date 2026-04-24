function inference = npd3qn_inference(scene, cfg, agents)
mission = struct('starts', cfg.starts, 'goals', cfg.goals);
[envState, observations] = npd3qn_env('reset', scene, cfg, mission);
numAgents = size(cfg.starts, 1);
paths = cell(numAgents, 1);
rawReward = zeros(numAgents, 1);
inferenceStepTime = zeros(cfg.maxStepsPerEpisode, 1);
for i = 1:numAgents
    paths{i} = envState.positions(i, :);
end
while any(envState.active) && envState.step < cfg.maxStepsPerEpisode
    stepTimer = tic;
    actions = ones(numAgents, 1);
    for i = 1:numAgents
        if ~envState.active(i)
            continue
        end
        q = npd3qn_net('predict', agents{i}.online, observations(:, i), cfg.useGPU);
        [~, actions(i)] = max(q);
    end
    [envState, observations, ~, ~, info] = npd3qn_env('step', scene, cfg, envState, actions);
    inferenceStepTime(envState.step) = toc(stepTimer);
    for i = 1:numAgents
        paths{i}(end+1, :) = envState.positions(i, :);
        rawReward(i) = rawReward(i) + info.rawRewards(i);
    end
end
inference = struct();
inference.paths = paths;
inference.success = envState.success;
inference.collision = envState.collision;
inference.steps = envState.step;
inference.rawReward = rawReward;
inference.pathLengths = compute_path_lengths(paths);
inference.totalPathLength = sum(inference.pathLengths);
validTimes = inferenceStepTime(1:max(envState.step, 1));
validTimes = validTimes(validTimes > 0);
if isempty(validTimes)
    inference.meanStepTimeMs = NaN;
else
    inference.meanStepTimeMs = 1000 * mean(validTimes);
end
end
function lengths = compute_path_lengths(paths)
numAgents = numel(paths);
lengths = zeros(numAgents, 1);
for i = 1:numAgents
    p = paths{i};
    if size(p, 1) < 2
        lengths(i) = 0;
    else
        lengths(i) = sum(vecnorm(diff(p, 1, 1), 2, 2));
    end
end
end
