## Viterbi algorithm parameters

# Deteriming Viterbi Parameters
beam_search = 0.7
bounding_region = 'wo'  # 'wo' weighted origin 'gb' look back m steps
look_back_m_steps = 3


SIMULATION_OPTION = {'beta': beam_search,
                     'bbox_type': bounding_region,
                     'm': look_back_m_steps}

