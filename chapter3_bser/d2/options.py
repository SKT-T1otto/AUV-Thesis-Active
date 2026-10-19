def performance_options(config):
    values = config.get('performance', {})
    if not isinstance(values, dict) or set(values) - {'learner_device', 'cpu_threads'}:
        raise ValueError('performance accepts only learner_device and cpu_threads')
    device = values.get('learner_device', 'cpu')
    threads = values.get('cpu_threads', 1)
    if device not in ('cpu', 'cuda'):
        raise ValueError('learner_device must explicitly be cpu or cuda')
    if type(threads) is not int or threads < 1:
        raise ValueError('cpu_threads must be a positive integer')
    return dict(learner_device=device, cpu_threads=threads)
