function scene = npd3qn_scene(cfg)
objPath = resolve_scene_file(cfg.scene.objFile, cfg.scene.objFileCandidates);
if isempty(objPath) || ~isfile(objPath)
    if isfield(cfg.scene, 'requireObj') && cfg.scene.requireObj
        error('Urban OBJ model is required. Put city.obj in the working directory or set cfg.scene.objFile.');
    end
    scene = build_procedural_scene(cfg.scene);
    return
end
[V, F] = read_obj_simple(objPath);
if isempty(V) || isempty(F)
    error('Failed to parse valid geometry from OBJ file: %s', objPath);
end
[V, F] = normalize_vertices(V, F, cfg.scene);
[xGrid, yGrid, H] = build_height_map(V, cfg.scene);
scene = struct();
scene.V = V;
scene.F = F;
scene.xRange = cfg.scene.xRange;
scene.yRange = cfg.scene.yRange;
scene.zRange = cfg.scene.zRange;
scene.xGrid = xGrid;
scene.yGrid = yGrid;
scene.H = H;
scene.heightMapStep = cfg.scene.heightMapStep;
scene.queryHeight = @(x, y) query_height(H, xGrid, yGrid, x, y);
end
function path = resolve_scene_file(primary, candidates)
path = '';
if nargin >= 1 && ~isempty(primary) && isfile(primary)
    path = primary;
    return
end
if nargin < 2 || isempty(candidates)
    return
end
for k = 1:numel(candidates)
    cand = candidates{k};
    if isfile(cand)
        path = cand;
        return
    end
end
end
function scene = build_procedural_scene(s)
[xGrid, yGrid] = meshgrid(s.xRange(1):s.heightMapStep:s.xRange(2), s.yRange(1):s.heightMapStep:s.yRange(2));
H = zeros(size(xGrid));
centers = [700 700; 1200 900; 1800 1200; 2400 850; 3000 1400; 900 2500; 1600 3000; 2500 2600; 3300 3200];
widths = [180 240 220 300 260 210 260 240 280];
heights = [180 320 260 420 360 240 380 300 450];
for k = 1:size(centers, 1)
    mask = abs(xGrid - centers(k, 1)) <= widths(k)/2 & abs(yGrid - centers(k, 2)) <= widths(k)/2;
    H(mask) = max(H(mask), heights(k));
end
scene = struct();
scene.V = zeros(0, 3);
scene.F = zeros(0, 3);
scene.xRange = s.xRange;
scene.yRange = s.yRange;
scene.zRange = s.zRange;
scene.xGrid = s.xRange(1):s.heightMapStep:s.xRange(2);
scene.yGrid = s.yRange(1):s.heightMapStep:s.yRange(2);
scene.H = H;
scene.heightMapStep = s.heightMapStep;
scene.queryHeight = @(x, y) query_height(H, scene.xGrid, scene.yGrid, x, y);
end
function [V, F] = read_obj_simple(filename)
V = zeros(0, 3);
F = zeros(0, 3);
fid = fopen(filename, 'r');
if fid < 0
    error('Unable to open OBJ file: %s', filename);
end
cleanup = onCleanup(@() fclose(fid));
faceCount = 0;
while ~feof(fid)
    line = strtrim(fgetl(fid));
    if ~ischar(line) || isempty(line)
        continue
    end
    if startsWith(line, 'v ')
        vals = sscanf(line(2:end), '%f');
        if numel(vals) >= 3
            V(end+1, :) = vals(1:3).';
        end
    elseif startsWith(line, 'f ')
        tokens = split(string(strtrim(line(2:end))));
        if numel(tokens) >= 3
            first = parse_face_index(tokens(1));
            for k = 2:numel(tokens)-1
                faceCount = faceCount + 1;
                F(faceCount, :) = [first, parse_face_index(tokens(k)), parse_face_index(tokens(k+1))];
            end
        end
    end
end
end
function idx = parse_face_index(token)
parts = split(token, '/');
idx = str2double(parts(1));
end
function [V, F] = normalize_vertices(V, F, s)
V(:, 1) = V(:, 1) - min(V(:, 1));
V(:, 2) = V(:, 2) - min(V(:, 2));
V(:, 3) = max(0, V(:, 3) - min(V(:, 3)));
spanX = max(max(V(:, 1)), eps);
spanY = max(max(V(:, 2)), eps);
spanZ = max(max(V(:, 3)), eps);
scaleXY = min(diff(s.xRange) / spanX, diff(s.yRange) / spanY);
V(:, 1) = s.xRange(1) + V(:, 1) * scaleXY;
V(:, 2) = s.yRange(1) + V(:, 2) * scaleXY;
V(:, 3) = s.zRange(1) + V(:, 3) * diff(s.zRange) / spanZ;
F = round(F);
end
function [xGrid, yGrid, H] = build_height_map(V, s)
step = s.heightMapStep;
xGrid = s.xRange(1):step:s.xRange(2);
yGrid = s.yRange(1):step:s.yRange(2);
nx = numel(xGrid);
ny = numel(yGrid);
ix = min(nx, max(1, floor((V(:, 1) - s.xRange(1)) / step) + 1));
iy = min(ny, max(1, floor((V(:, 2) - s.yRange(1)) / step) + 1));
ind = sub2ind([ny, nx], iy, ix);
H = accumarray(ind, V(:, 3), [ny * nx, 1], @max, 0);
H = reshape(H, [ny, nx]);
K = ones(5, 5, 'double') / 25;
H = conv2(H, K, 'same');
end
function h = query_height(H, xGrid, yGrid, x, y)
x = min(max(x, xGrid(1)), xGrid(end));
y = min(max(y, yGrid(1)), yGrid(end));
[X, Y] = meshgrid(xGrid, yGrid);
h = interp2(X, Y, H, x, y, 'linear', 0);
end
