
def algorithm_label(id, config):
    # hier kannst du beliebige config settings auslesen
    algorithm = config['experiment.simple']['algorithm'].rsplit('.', 1)[1]
    model = config['algorithm']['model'].rsplit('.', 1)[1]
    return f"{algorithm}, {model}"  # return label