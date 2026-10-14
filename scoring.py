def score_episode(landed, obs, fires):
    if not landed:
        return 0.0
    pts = 50.0
    pts += 20 * max(0.0, 1 - abs(obs[0]) / 0.5)      # position
    pts += 10 * (obs[6] == 1 and obs[7] == 1)        # legs
    pts += 10 * max(0.0, 1 - abs(obs[4]) / 0.5)      # upright
    pts += 10 * max(0.0, 1 - fires / 400)            # fuel
    return pts