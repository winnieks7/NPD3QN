function varargout = npd3qn_net(mode, varargin)
switch lower(mode)
    case 'init'
        cfg = varargin{1};
        agent = init_agent(cfg);
        varargout = {agent};
    case 'predict'
        [q, value, advantage] = predict_q(varargin{:});
        varargout = {q, value, advantage};
    case 'train'
        [agent, lossValue, tdAbs] = train_agent(varargin{:});
        varargout = {agent, lossValue, tdAbs};
    case 'copy'
        target = varargin{1};
        online = varargin{2};
        target = copy_target(target, online);
        varargout = {target};
    otherwise
        error('Unsupported mode.');
end
end
function agent = init_agent(cfg)
lgraph = layerGraph();
common = [
    featureInputLayer(cfg.obsDim, Normalization="none", Name="state")
    fullyConnectedLayer(cfg.hiddenWidths(1), Name="fc1")
    reluLayer(Name="relu1")
    fullyConnectedLayer(cfg.hiddenWidths(2), Name="fc2")
    reluLayer(Name="relu2")
];
valueHead = [
    fullyConnectedLayer(cfg.duelingWidth, Name="value_fc1")
    reluLayer(Name="value_relu1")
    fullyConnectedLayer(1, Name="value_out")
];
advHead = [
    fullyConnectedLayer(cfg.duelingWidth, Name="adv_fc1")
    reluLayer(Name="adv_relu1")
    fullyConnectedLayer(cfg.numActions, Name="adv_out")
];
lgraph = addLayers(lgraph, common);
lgraph = addLayers(lgraph, valueHead);
lgraph = addLayers(lgraph, advHead);
lgraph = connectLayers(lgraph, "relu2", "value_fc1");
lgraph = connectLayers(lgraph, "relu2", "adv_fc1");
online = dlnetwork(lgraph);
target = dlnetwork(lgraph);
target = copy_target(target, online);
trailingAvg = cell(size(online.Learnables, 1), 1);
trailingAvgSq = cell(size(online.Learnables, 1), 1);
for i = 1:size(online.Learnables, 1)
    trailingAvg{i} = zeros(size(online.Learnables.Value{i}), 'like', online.Learnables.Value{i});
    trailingAvgSq{i} = zeros(size(online.Learnables.Value{i}), 'like', online.Learnables.Value{i});
end
agent = struct('online', online, 'target', target, 'trailingAvg', {trailingAvg}, 'trailingAvgSq', {trailingAvgSq});
end
function [q, value, advantage] = predict_q(net, observation, useGPU)
x = single(observation);
if isvector(x)
    x = reshape(x, [], 1);
end
dlX = dlarray(x, 'CB');
if useGPU
    dlX = gpuArray(dlX);
end
[value, advantage] = forward(net, dlX, Outputs=["value_out", "adv_out"]);
q = value + (advantage - mean(advantage, 1));
q = gather(extractdata(q));
value = gather(extractdata(value));
advantage = gather(extractdata(advantage));
end
function [agent, lossValue, tdAbs] = train_agent(agent, batch, cfg, iteration, useGPU)
B = numel(batch.actions);
obsBatch = single(cat(2, batch.states{:}));
nextObsBatch = single(cat(2, batch.nextStates{:}));
actions = single(reshape(batch.actions, 1, []));
returns = single(reshape(batch.returns, 1, []));
dones = single(reshape(batch.dones, 1, []));
gammaPows = single(reshape(batch.gammaPows, 1, []));
weights = single(reshape(batch.isWeights, 1, []));
dlX = dlarray(obsBatch, 'CB');
dlXNext = dlarray(nextObsBatch, 'CB');
if useGPU
    dlX = gpuArray(dlX);
    dlXNext = gpuArray(dlXNext);
end
qNextOnline = compute_q(agent.online, dlXNext);
qNextTarget = compute_q(agent.target, dlXNext);
qNextOnline = gather(extractdata(qNextOnline));
qNextTarget = gather(extractdata(qNextTarget));
targets = zeros(1, B, 'single');
for i = 1:B
    if dones(i) > 0
        targets(i) = returns(i);
    else
        [~, bestAct] = max(qNextOnline(:, i));
        targets(i) = returns(i) + gammaPows(i) * qNextTarget(bestAct, i);
    end
end
[loss, gradients, tdAbs] = dlfeval(@model_gradients, agent.online, dlX, actions, targets, weights);
for p = 1:size(agent.online.Learnables, 1)
    [agent.online.Learnables.Value{p}, agent.trailingAvg{p}, agent.trailingAvgSq{p}] = adamupdate( ...
        agent.online.Learnables.Value{p}, gradients.Value{p}, agent.trailingAvg{p}, agent.trailingAvgSq{p}, ...
        iteration, cfg.learningRate, 0.9, 0.999, 1e-8);
end
lossValue = double(gather(extractdata(loss)));
tdAbs = double(gather(extractdata(tdAbs)));
end
function target = copy_target(target, online)
for i = 1:size(online.Learnables, 1)
    target.Learnables.Value{i} = online.Learnables.Value{i};
end
end
function q = compute_q(net, dlX)
[value, advantage] = forward(net, dlX, Outputs=["value_out", "adv_out"]);
q = value + (advantage - mean(advantage, 1));
end
function [loss, gradients, tdAbs] = model_gradients(net, dlX, actions, targets, weights)
q = compute_q(net, dlX);
B = size(q, 2);
actionIdx = local_action_indices(actions, B, size(q, 1));
linearIdx = sub2ind(size(q), actionIdx, 1:B);
selected = reshape(q(linearIdx), 1, []);
targetDl = dlarray(targets);
weightDl = dlarray(weights);
td = targetDl - selected;
loss = mean(weightDl .* td.^2, 'all');
gradients = dlgradient(loss, net.Learnables);
tdAbs = abs(td);
end
function actionIdx = local_action_indices(actions, batchSize, numActions)
if isa(actions, 'dlarray')
    actionIdx = extractdata(actions);
else
    actionIdx = actions;
end
if isa(actionIdx, 'gpuArray')
    actionIdx = gather(actionIdx);
end
actionIdx = reshape(double(actionIdx), 1, []);
if numel(actionIdx) ~= batchSize
    error('Action batch size mismatch.');
end
actionIdx = round(actionIdx);
actionIdx = max(1, min(numActions, actionIdx));
end
