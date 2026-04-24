function varargout = npd3qn_env(mode, scene, cfg, varargin)
switch lower(mode)
    case 'reset'
        [envState, observations] = reset_env(scene, cfg, varargin{:});
        varargout = {envState, observations};
    case 'step'
        [envState, observations, rewards, dones, info] = step_env(scene, cfg, varargin{:});
        varargout = {envState, observations, rewards, dones, info};
    otherwise
        error('Unsupported mode.');
end
end
function [envState, observations] = reset_env(scene, cfg, mission)
positions = round(mission.starts);
goals = round(mission.goals);
numAgents = size(positions, 1);
envState = struct();
envState.positions = positions;
envState.goals = goals;
envState.active = true(numAgents, 1);
envState.success = false(numAgents, 1);
envState.collision = false(numAgents, 1);
envState.step = 0;
observations = build_observations(scene, cfg, envState);
end
function [envState, observations, rewards, dones, info] = step_env(scene, cfg, envState, actions)
numAgents = size(envState.positions, 1);
actionTable = get_action_table();
currentPos = envState.positions;
goals = envState.goals;
proposed = currentPos;
collisionObs = false(numAgents, 1);
collisionObsPath = false(numAgents, 1);
collisionUav = false(numAgents, 1);
collisionUavPath = false(numAgents, 1);
collisionSwap = false(numAgents, 1);
outOfBounds = false(numAgents, 1);
reachGoal = false(numAgents, 1);
rawRewards = zeros(numAgents, 1);
windVectors = zeros(numAgents, 3);
windAlignment = nan(numAgents, 1);
actionMoves = zeros(numAgents, 3);
obstacleNearest = inf(numAgents, 1);
uavNearest = inf(numAgents, 1);
distancePrev = inf(numAgents, 1);
distanceCurr = inf(numAgents, 1);
pathStartPhysical = currentPos * cfg.gridResolution;
pathEndPhysical = pathStartPhysical;
segmentDistance = inf(numAgents, numAgents);
for i = 1:numAgents
    if ~envState.active(i)
        continue
    end
    move = actionTable(actions(i), :);
    wind = wind_field(scene, cfg, currentPos(i, :));
    windVectors(i, :) = wind;
    actionMoves(i, :) = move;
    windAlignment(i) = wind_action_alignment(move, wind);
    currentPhysical = currentPos(i, :) * cfg.gridResolution;
    nextPhysical = currentPhysical + cfg.gridResolution * move + cfg.dt * wind;
    pathStartPhysical(i, :) = currentPhysical;
    pathEndPhysical(i, :) = nextPhysical;
    candidate = round(nextPhysical / cfg.gridResolution);
    [pathObs, pathBounds] = path_environment_collision(scene, cfg, currentPhysical, nextPhysical);
    if pathBounds || candidate(1) < scene.xRange(1) || candidate(1) > scene.xRange(2) || ...
       candidate(2) < scene.yRange(1) || candidate(2) > scene.yRange(2) || ...
       candidate(3) < scene.zRange(1) || candidate(3) > scene.zRange(2)
        outOfBounds(i) = true;
        candidate = currentPos(i, :);
    end
    if pathObs
        collisionObsPath(i) = true;
        collisionObs(i) = true;
        candidate = currentPos(i, :);
    end
    if ~outOfBounds(i) && ~collisionObs(i)
        terrainHeight = scene.queryHeight(candidate(1), candidate(2));
        if candidate(3) <= terrainHeight
            collisionObs(i) = true;
            candidate = currentPos(i, :);
        end
    end
    proposed(i, :) = candidate;
end
for i = 1:numAgents-1
    if ~envState.active(i)
        continue
    end
    for j = i+1:numAgents
        if ~envState.active(j)
            continue
        end
        terminalDistance = norm(pathEndPhysical(i, :) - pathEndPhysical(j, :), 2);
        segDistance = segment_distance_3d(pathStartPhysical(i, :), pathEndPhysical(i, :), pathStartPhysical(j, :), pathEndPhysical(j, :));
        segmentDistance(i, j) = segDistance;
        segmentDistance(j, i) = segDistance;
        swapped = paths_swapped(currentPos(i, :), proposed(i, :), currentPos(j, :), proposed(j, :), pathStartPhysical(i, :), pathEndPhysical(i, :), pathStartPhysical(j, :), pathEndPhysical(j, :), cfg);
        if terminalDistance < cfg.dSafe || segDistance < cfg.dSafe || swapped
            collisionUav(i) = true;
            collisionUav(j) = true;
            if segDistance < cfg.dSafe
                collisionUavPath(i) = true;
                collisionUavPath(j) = true;
            end
            if swapped
                collisionSwap(i) = true;
                collisionSwap(j) = true;
            end
        end
    end
end
finalPlanned = proposed;
for i = 1:numAgents
    if collisionObs(i) || collisionUav(i) || outOfBounds(i)
        finalPlanned(i, :) = currentPos(i, :);
    end
end
for i = 1:numAgents
    if ~envState.active(i)
        continue
    end
    nextPos = finalPlanned(i, :);
    dPrev = norm(currentPos(i, :) - goals(i, :), 2);
    dCurr = norm(nextPos - goals(i, :), 2);
    distancePrev(i) = dPrev;
    distanceCurr(i) = dCurr;
    if ~(collisionObs(i) || collisionUav(i) || outOfBounds(i)) && dCurr <= cfg.dSafe
        nextPos = goals(i, :);
        dCurr = 0;
        distanceCurr(i) = dCurr;
        reachGoal(i) = true;
    end
    obstacleDistances = obstacle_distances(scene, cfg, nextPos);
    uavDistances = inter_uav_distances(finalPlanned, i, envState.active);
    obstacleNearest(i) = local_nearest_distance(obstacleDistances, cfg.sensingRange);
    uavNearest(i) = local_nearest_distance(uavDistances, cfg.sensingRange);
    rawRewards(i) = immediate_reward(cfg, currentPos(i, :), nextPos, goals(i, :), actions(i), ...
        wind_field(scene, cfg, currentPos(i, :)), dPrev, dCurr, obstacleDistances, uavDistances, ...
        collisionObs(i) || outOfBounds(i), collisionUav(i), reachGoal(i));
    envState.positions(i, :) = nextPos;
    if reachGoal(i)
        envState.success(i) = true;
        envState.active(i) = false;
    elseif collisionObs(i) || collisionUav(i) || outOfBounds(i)
        envState.collision(i) = true;
        envState.active(i) = false;
    end
end
envState.step = envState.step + 1;
rewards = rawRewards;
dones = ~envState.active;
if envState.step >= cfg.maxStepsPerEpisode
    dones = true(numAgents, 1);
    envState.active(:) = false;
end
observations = build_observations(scene, cfg, envState);
info = struct();
info.rawRewards = rawRewards;
info.collisionObs = collisionObs | outOfBounds;
info.collisionObsPath = collisionObsPath;
info.collisionUav = collisionUav;
info.collisionUavPath = collisionUavPath;
info.collisionSwap = collisionSwap;
info.reachGoal = reachGoal;
info.localWind = windVectors;
info.actionMove = actionMoves;
info.windAlignment = windAlignment;
info.isTailwind = windAlignment > cfg.per.tailwindCosThreshold;
info.distancePrev = distancePrev;
info.distanceCurr = distanceCurr;
info.isCloserToGoal = distanceCurr < distancePrev - cfg.per.progressTol;
info.obstacleDistance = obstacleNearest;
info.uavDistance = uavNearest;
info.pathStartPhysical = pathStartPhysical;
info.pathEndPhysical = pathEndPhysical;
info.segmentDistance = segmentDistance;
info.normalizedRewards = [];
end
function observations = build_observations(scene, cfg, envState)
numAgents = size(envState.positions, 1);
observations = zeros(cfg.obsDim, numAgents, 'single');
span = [diff(scene.xRange), diff(scene.yRange), diff(scene.zRange)];
for i = 1:numAgents
    if ~envState.active(i)
        continue
    end
    pos = envState.positions(i, :);
    goal = envState.goals(i, :);
    goalDelta = (goal - pos) ./ max(span, 1);
    dObs = min(local_nearest_distance(obstacle_distances(scene, cfg, pos), cfg.sensingRange), cfg.sensingRange);
    dUav = min(local_nearest_distance(inter_uav_distances(envState.positions, i, envState.active), cfg.sensingRange), cfg.sensingRange);
    wind = wind_field(scene, cfg, pos);
    windSpeed = min(norm(wind, 2), cfg.wind.maxSpeed);
    obs = zeros(9, 1);
    obs(1:3) = max(-1, min(1, goalDelta(:)));
    obs(4) = max(-1, min(1, 2 * dObs / cfg.sensingRange - 1));
    obs(5) = max(-1, min(1, 2 * dUav / cfg.sensingRange - 1));
    obs(6:8) = max(-1, min(1, wind(:) / cfg.wind.maxSpeed));
    obs(9) = max(-1, min(1, 2 * windSpeed / cfg.wind.maxSpeed - 1));
    observations(:, i) = single(obs);
end
end
function reward = immediate_reward(cfg, prevPos, nextPos, goalPos, actionIdx, wind, dPrev, dCurr, obstacleDistances, uavDistances, collisionObs, collisionUav, reachGoal)
if reachGoal
    reward = cfg.goalReward;
    return
end
Rdist = cfg.omega1 * (dPrev - dCurr) - cfg.cStep;
actionTable = get_action_table();
move = actionTable(actionIdx, :);
if norm(move, 2) < eps || norm(wind, 2) < eps
    Rwind = 0;
else
    Rwind = cfg.omega2 * dot(move / norm(move, 2), wind / norm(wind, 2));
end
Robs = -cfg.cObs * double(collisionObs) - cfg.omega3 * safety_penalty_sum(obstacleDistances, cfg.dSafe);
Ruav = -cfg.cUav * double(collisionUav) - cfg.omega4 * safety_penalty_sum(uavDistances, cfg.dSafe);
reward = Rdist + Rwind + Robs + Ruav;
end
function total = safety_penalty_sum(distances, dSafe)
if isempty(distances)
    total = 0;
    return
end
mask = distances < dSafe;
if ~any(mask)
    total = 0;
    return
end
total = sum(1 ./ (distances(mask) + 0.1));
end
function [hitObstacle, hitBounds] = path_environment_collision(scene, cfg, p0, p1)
hitObstacle = false;
hitBounds = false;
stepSize = cfg.collision.pathSampleStep;
if isempty(stepSize) || ~isfinite(stepSize) || stepSize <= 0
    stepSize = max(cfg.gridResolution * 0.5, eps);
end
nSamples = max(2, ceil(norm(p1 - p0, 2) / stepSize) + 1);
for k = 1:nSamples
    if nSamples == 1
        t = 1;
    else
        t = (k - 1) / (nSamples - 1);
    end
    p = p0 + t * (p1 - p0);
    pos = p / cfg.gridResolution;
    if pos(1) < scene.xRange(1) || pos(1) > scene.xRange(2) || ...
       pos(2) < scene.yRange(1) || pos(2) > scene.yRange(2) || ...
       pos(3) < scene.zRange(1) || pos(3) > scene.zRange(2)
        hitBounds = true;
        return
    end
    terrainHeight = scene.queryHeight(pos(1), pos(2));
    if pos(3) <= terrainHeight + cfg.collision.obstacleClearance
        hitObstacle = true;
        return
    end
end
end
function swapped = paths_swapped(startI, endI, startJ, endJ, p0I, p1I, p0J, p1J, cfg)
gridTol = cfg.collision.swapTolerance;
if isempty(gridTol) || ~isfinite(gridTol) || gridTol < 0
    gridTol = 0;
end
swapGrid = norm(endI - startJ, 2) <= gridTol && norm(endJ - startI, 2) <= gridTol;
swapPhysical = norm(p1I - p0J, 2) < cfg.dSafe && norm(p1J - p0I, 2) < cfg.dSafe;
swapped = swapGrid || swapPhysical;
end
function d = segment_distance_3d(p1, q1, p2, q2)
u = q1 - p1;
v = q2 - p2;
w = p1 - p2;
a = dot(u, u);
b = dot(u, v);
c = dot(v, v);
d0 = dot(u, w);
e = dot(v, w);
D = a * c - b * b;
small = 1e-12;
sN = 0;
sD = D;
tN = 0;
tD = D;
if D < small
    sN = 0;
    sD = 1;
    tN = e;
    tD = c;
else
    sN = b * e - c * d0;
    tN = a * e - b * d0;
    if sN < 0
        sN = 0;
        tN = e;
        tD = c;
    elseif sN > sD
        sN = sD;
        tN = e + b;
        tD = c;
    end
end
if tN < 0
    tN = 0;
    if -d0 < 0
        sN = 0;
    elseif -d0 > a
        sN = sD;
    else
        sN = -d0;
        sD = a;
    end
elseif tN > tD
    tN = tD;
    if (-d0 + b) < 0
        sN = 0;
    elseif (-d0 + b) > a
        sN = sD;
    else
        sN = -d0 + b;
        sD = a;
    end
end
if abs(sN) < small
    sc = 0;
else
    sc = sN / sD;
end
if abs(tN) < small
    tc = 0;
else
    tc = tN / tD;
end
dP = w + sc * u - tc * v;
d = norm(dP, 2);
end
function distances = obstacle_distances(scene, cfg, pos)
xSpan = scene.xGrid;
ySpan = scene.yGrid;
step = scene.heightMapStep;
radius = cfg.sensingRange;
ix = max(1, floor((pos(1) - radius - xSpan(1)) / step) + 1):min(numel(xSpan), ceil((pos(1) + radius - xSpan(1)) / step) + 1);
iy = max(1, floor((pos(2) - radius - ySpan(1)) / step) + 1):min(numel(ySpan), ceil((pos(2) + radius - ySpan(1)) / step) + 1);
if isempty(ix) || isempty(iy)
    distances = inf(0, 1);
    return
end
[X, Y] = meshgrid(xSpan(ix), ySpan(iy));
Hs = scene.H(iy, ix);
occupied = Hs > 0;
if ~any(occupied(:))
    distances = inf(0, 1);
    return
end
dx = X(occupied) - pos(1);
dy = Y(occupied) - pos(2);
dz = max(Hs(occupied) - pos(3), 0);
distances = sqrt(dx.^2 + dy.^2 + dz.^2);
distances = distances(:);
end
function distances = inter_uav_distances(positions, idx, active)
others = active;
others(idx) = false;
if ~any(others)
    distances = inf(0, 1);
    return
end
distances = vecnorm(positions(others, :) - positions(idx, :), 2, 2);
end
function d = local_nearest_distance(distances, fallback)
if isempty(distances)
    d = fallback;
    return
end
d = min(distances(:));
if ~isfinite(d)
    d = fallback;
end
end
function alignment = wind_action_alignment(move, wind)
if norm(move, 2) < eps || norm(wind, 2) < eps
    alignment = 0;
else
    alignment = dot(move / norm(move, 2), wind / norm(wind, 2));
end
end
function wind = wind_field(scene, cfg, pos)
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
function actionTable = get_action_table()
persistent A
if isempty(A)
    A = zeros(26, 3);
    idx = 0;
    for dx = -1:1
        for dy = -1:1
            for dz = -1:1
                if dx == 0 && dy == 0 && dz == 0
                    continue
                end
                idx = idx + 1;
                A(idx, :) = [dx, dy, dz];
            end
        end
    end
end
actionTable = A;
end
