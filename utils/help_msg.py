def help_msg():
    msg = """
    Usage: python main.py [options]
    Options:
        --retain <model_name> Force retrain the specified model
        --retrain all Force retrain all models
        --retrain Retrain all models
        --generate_data Generate/ get the training and testing data
        """
    print(msg)