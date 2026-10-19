PROTOCOL = 'd2_v1'
LEGACY = 'legacy_bser'

def planner_protocol(config):
    value = config.get('planner_protocol', LEGACY)
    if value not in (LEGACY, PROTOCOL):
        raise ValueError('unknown planner_protocol: ' + str(value))
    return value

def enabled(config):
    return planner_protocol(config) == PROTOCOL
