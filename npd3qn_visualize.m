function figInfo = npd3qn_visualize(scene, cfg, training, inference, seedIndex)
if nargin == 0
    [scene, cfg, training, inference, seedIndex] = local_load_default_inputs();
elseif nargin < 4
    error('npd3qn_visualize requires either no inputs or scene, cfg, training, and inference inputs.');
elseif nargin < 5
    seedIndex = [];
end
if ~isfield(cfg, 'figureDir') || isempty(cfg.figureDir)
    cfg.figureDir = 'npd3qn_figures';
end
if ~isfield(cfg, 'seed') || isempty(cfg.seed)
    cfg.seed = 0;
end
if ~isfield(inference, 'pathLengths') || isempty(inference.pathLengths)
    inference.pathLengths = local_compute_path_lengths(inference.paths);
end
if ~isfolder(cfg.figureDir)
    mkdir(cfg.figureDir);
end
colors = [0.90 0.10 0.10; 0.95 0.55 0.10; 0.10 0.35 0.90; 0.20 0.70 0.30; 0.70 0.20 0.70];
fig1 = figure('Color', 'w', 'Position', [80 60 1280 900], 'Name', sprintf('Path Planning Seed %d', cfg.seed));
ax = axes('Parent', fig1);
hold(ax, 'on');
local_plot_scene(ax, scene);
hWind = local_plot_wind_field(ax, scene, cfg);
pathHandles = gobjects(numel(inference.paths), 1);
for i = 1:numel(inference.paths)
    c = colors(mod(i-1, size(colors,1)) + 1, :);
    p = inference.paths{i};
    pathHandles(i) = plot3(ax, p(:,1), p(:,2), p(:,3), '-', 'LineWidth', 2.6, 'Color', c);
    scatter3(ax, cfg.starts(i,1), cfg.starts(i,2), cfg.starts(i,3), 80, c, 'filled', 'o');
    scatter3(ax, cfg.goals(i,1), cfg.goals(i,2), cfg.goals(i,3), 100, c, 'filled', '^');
end
xlabel(ax, 'X (m)');
ylabel(ax, 'Y (m)');
zlabel(ax, 'Z (m)');
title(ax, sprintf('Figure 18 Style Path Planning Result with Urban Scene and Wind Field (Seed %d)', cfg.seed), 'FontSize', 14, 'FontWeight', 'bold');
view(ax, 38, 28);
grid(ax, 'on');
axis(ax, 'equal');
xlim(ax, scene.xRange);
ylim(ax, scene.yRange);
zlim(ax, scene.zRange);
camlight(ax, 'headlight');
lighting(ax, 'gouraud');
labels = cell(1, numel(inference.paths));
for i = 1:numel(inference.paths)
    if inference.success(i)
        s = 'Success';
    elseif inference.collision(i)
        s = 'Collision';
    else
        s = 'Incomplete';
    end
    labels{i} = sprintf('UAV-%d | %s | L=%.1f', i, s, inference.pathLengths(i));
end
if isgraphics(hWind)
    legendHandles = [hWind; pathHandles];
    legendLabels = [{'Wind field'}, labels];
else
    legendHandles = pathHandles;
    legendLabels = labels;
end
legend(ax, legendHandles, legendLabels, 'Location', 'northeastoutside');
pathPng = fullfile(cfg.figureDir, sprintf('figure18_style_path_seed_%d.png', cfg.seed));
pathFig = fullfile(cfg.figureDir, sprintf('figure18_style_path_seed_%d.fig', cfg.seed));
local_export_figure(fig1, pathPng);
savefig(fig1, pathFig);
fig2 = figure('Color', 'w', 'Position', [120 100 1100 520], 'Name', sprintf('Reward Convergence Seed %d', cfg.seed));
plot(training.episodeReward, 'LineWidth', 1.8);
hold on;
window = min(50, numel(training.episodeReward));
if window > 1
    smoothCurve = movmean(training.episodeReward, window);
    plot(smoothCurve, 'LineWidth', 2.4);
    legend({'Episode reward', sprintf('Moving average (%d)', window)}, 'Location', 'best');
else
    legend({'Episode reward'}, 'Location', 'best');
end
xlabel('Episode');
ylabel('Reward');
title(sprintf('Figure 19 Style Reward Convergence (Seed %d)', cfg.seed), 'FontSize', 14, 'FontWeight', 'bold');
grid on;
rewardPng = fullfile(cfg.figureDir, sprintf('figure19_style_reward_seed_%d.png', cfg.seed));
rewardFig = fullfile(cfg.figureDir, sprintf('figure19_style_reward_seed_%d.fig', cfg.seed));
local_export_figure(fig2, rewardPng);
savefig(fig2, rewardFig);
figInfo = struct('pathPng', pathPng, 'pathFig', pathFig, 'rewardPng', rewardPng, 'rewardFig', rewardFig);
if ~isempty(seedIndex)
    figInfo.seedIndex = seedIndex;
end
end
function h = local_plot_scene(ax, scene)
h = gobjects(0);
if isfield(scene, 'V') && isfield(scene, 'F') && ~isempty(scene.V) && ~isempty(scene.F)
    h = patch('Parent', ax, 'Vertices', scene.V, 'Faces', scene.F, 'FaceColor', [0.75 0.75 0.78], 'EdgeColor', 'none', 'FaceAlpha', 0.35);
elseif isfield(scene, 'H') && ~isempty(scene.H) && isfield(scene, 'xGrid') && isfield(scene, 'yGrid')
    [X, Y] = meshgrid(scene.xGrid, scene.yGrid);
    h = surf(ax, X, Y, scene.H, 'EdgeColor', 'none', 'FaceColor', [0.75 0.75 0.78], 'FaceAlpha', 0.35);
end
end
function h = local_plot_wind_field(ax, scene, cfg)
countXY = local_get_cfg_value(cfg, {'windVisualization', 'countXY'}, 9);
countZ = local_get_cfg_value(cfg, {'windVisualization', 'countZ'}, 4);
lengthScale = local_get_cfg_value(cfg, {'windVisualization', 'lengthScale'}, 12.0);
terrainClearance = local_get_cfg_value(cfg, {'windVisualization', 'terrainClearance'}, 40.0);
lineWidth = local_get_cfg_value(cfg, {'windVisualization', 'lineWidth'}, 1.1);
xSamples = linspace(scene.xRange(1), scene.xRange(2), countXY);
ySamples = linspace(scene.yRange(1), scene.yRange(2), countXY);
zSamples = linspace(scene.zRange(1) + terrainClearance, scene.zRange(2), countZ);
xs = [];
ys = [];
zs = [];
us = [];
vs = [];
ws = [];
for ix = 1:numel(xSamples)
    for iy = 1:numel(ySamples)
        terrainHeight = local_scene_height(scene, xSamples(ix), ySamples(iy));
        for iz = 1:numel(zSamples)
            z = max(zSamples(iz), terrainHeight + terrainClearance);
            if z > scene.zRange(2)
                continue
            end
            wind = local_wind_field(scene, cfg, [xSamples(ix), ySamples(iy), z]);
            xs(end+1, 1) = xSamples(ix);
            ys(end+1, 1) = ySamples(iy);
            zs(end+1, 1) = z;
            us(end+1, 1) = wind(1) * lengthScale;
            vs(end+1, 1) = wind(2) * lengthScale;
            ws(end+1, 1) = wind(3) * lengthScale;
        end
    end
end
if isempty(xs)
    h = gobjects(0);
    return
end
h = quiver3(ax, xs, ys, zs, us, vs, ws, 0, 'Color', [0.15 0.45 0.95], 'LineWidth', lineWidth, 'MaxHeadSize', 0.45);
end
function h = local_scene_height(scene, x, y)
if isfield(scene, 'queryHeight') && ~isempty(scene.queryHeight)
    h = scene.queryHeight(x, y);
elseif isfield(scene, 'H') && isfield(scene, 'xGrid') && isfield(scene, 'yGrid') && ~isempty(scene.H)
    [X, Y] = meshgrid(scene.xGrid, scene.yGrid);
    h = interp2(X, Y, scene.H, x, y, 'linear', 0);
else
    h = 0;
end
if ~isfinite(h)
    h = 0;
end
end
function wind = local_wind_field(scene, cfg, pos)
x = pos(1);
y = pos(2);
z = pos(3);
xr = max(diff(scene.xRange), 1);
yr = max(diff(scene.yRange), 1);
zr = max(diff(scene.zRange), 1);
xn = 2 * (x - mean(scene.xRange)) / xr;
yn = 2 * (y - mean(scene.yRange)) / yr;
zn = (z - scene.zRange(1)) / zr;
u = cfg.wind.baseSpeed + 1.6 * sin(pi * yn) + 1.2 * cos(pi * xn) + 0.8 * zn;
v = 1.0 * cos(pi * xn) - 0.8 * sin(pi * yn);
w = cfg.wind.verticalScale * sin(pi * xn) .* cos(pi * yn);
wind = [u, v, w];
speed = norm(wind, 2);
if speed < eps
    wind = [cfg.wind.minSpeed, 0, 0];
    speed = norm(wind, 2);
end
if speed < cfg.wind.minSpeed
    wind = wind * (cfg.wind.minSpeed / speed);
elseif speed > cfg.wind.maxSpeed
    wind = wind * (cfg.wind.maxSpeed / speed);
end
end
function value = local_get_cfg_value(cfg, fields, defaultValue)
value = defaultValue;
if isempty(fields)
    return
end
current = cfg;
for k = 1:numel(fields)
    name = fields{k};
    if ~isstruct(current) || ~isfield(current, name) || isempty(current.(name))
        return
    end
    current = current.(name);
end
value = current;
end
function [scene, cfg, training, inference, seedIndex] = local_load_default_inputs()
cfg = npd3qn_config();
seedIndex = 1;
resultFile = cfg.outputFile;
if exist(resultFile, 'file') ~= 2
    resultFile = fullfile(fileparts(mfilename('fullpath')), cfg.outputFile);
end
if exist(resultFile, 'file') ~= 2
    error('Default result file was not found. Run main_npd3qn first or call npd3qn_visualize with scene, cfg, training, and inference.');
end
loaded = load(resultFile, 'cfg', 'results', 'scene');
if isfield(loaded, 'cfg')
    cfg = loaded.cfg;
end
if isfield(loaded, 'scene')
    scene = loaded.scene;
else
    scene = npd3qn_scene(cfg);
end
if ~isfield(loaded, 'results') || isempty(loaded.results)
    error('Default result file does not contain visualization results.');
end
result = loaded.results{seedIndex};
training = result.training;
inference = result.inference;
if isfield(result, 'seed')
    cfg.seed = result.seed;
end
end
function local_export_figure(figHandle, fileName)
if exist('exportgraphics', 'file') == 2
    exportgraphics(figHandle, fileName, 'Resolution', 220);
else
    print(figHandle, fileName, '-dpng', '-r220');
end
end
function lengths = local_compute_path_lengths(paths)
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
