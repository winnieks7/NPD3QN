function results = main_npd3qn()
cfg = npd3qn_config();
scene = npd3qn_scene(cfg);
numSeeds = numel(cfg.seeds);
results = cell(numSeeds, 1);
for k = 1:numSeeds
    runCfg = cfg;
    runCfg.seed = cfg.seeds(k);
    runCfg.totalTrainingSteps = runCfg.maxEpisodes * runCfg.maxStepsPerEpisode;
    runCfg.epsilonDecaySteps = max(1, round(0.2 * runCfg.totalTrainingSteps));
    rng(runCfg.seed, 'twister');
    fprintf('Seed %d/%d\n', k, numSeeds);
    training = npd3qn_training(scene, runCfg);
    inference = npd3qn_inference(scene, runCfg, training.bestAgents);
    figInfo = [];
    if isfield(runCfg, 'figureDir') && ~isempty(runCfg.figureDir)
        figInfo = npd3qn_visualize(scene, runCfg, training, inference, k);
    end
    results{k} = struct('seed', runCfg.seed, 'training', training, 'inference', inference, 'figures', figInfo);
end
save(cfg.outputFile, 'cfg', 'results', 'scene', '-v7.3');
end
