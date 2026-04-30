from vla_project.data import constants as C


def test_ranges_dont_overlap():
    soft = set(range(C.SOFT_PROMPT_BEGIN_IDX,
                     C.SOFT_PROMPT_BEGIN_IDX + C.NUM_SOFT_PROMPT_TOKENS))
    wrist = set(range(C.WRIST_PLACEHOLDER_BEGIN_IDX,
                      C.WRIST_PLACEHOLDER_BEGIN_IDX + C.NUM_WRIST_TOKENS))
    action = set(range(C.ACTION_TOKEN_BEGIN_IDX,
                       C.ACTION_TOKEN_BEGIN_IDX + C.NUM_ACTION_TOKENS))
    assert soft.isdisjoint(wrist)
    assert soft.isdisjoint(action)
    assert wrist.isdisjoint(action)


def test_image_soft_token_distinct():
    assert C.IMAGE_SOFT_TOKEN_ID not in range(
        C.SOFT_PROMPT_BEGIN_IDX,
        C.SOFT_PROMPT_BEGIN_IDX + C.NUM_SOFT_PROMPT_TOKENS,
    )
