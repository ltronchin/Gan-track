def print_section(print_string):
    print("\n"*2)
    print("="*100)
    print("working on:", print_string)
    print("="*100)


def print_processing(print_string):
    print(">> processing", print_string)


def print_separator():
    print("\n")
    print("-"*100)


def print_training_summary(model, config):

    print("\n"*2)
    print("="*100)
    print("TRAINING SUMMARY")
    print("="*100)
    print("project:", config["project"])
    print("model: {} \t dimension: {}".format(
        config["model"], config["model_dim"]))
    try:
        print("model input: {} \t model output: {}".format(
            model.input.get_shape().as_list(), model.output.get_shape().as_list()))
    except:
        pass
    print("-"*100)
    print("number of train: {} \t val: {} \t test: {}".format(config["n_training_patient"],
                                                              config["n_validation_patient"],
                                                              config["n_testing_patient"]))

    print("training on labels:", config["labels"])
    print("-"*100)
    print("initial learning rate: {} \t learning rate drop: {}".format(
        config["initial_learning_rate"], config["learning_rate_drop"]))
    print("-"*100)
    print("data file:", config["data_file"])
    print("model file:", config["model_file"])
    print("training file:", config["training_file"])
    print("validation file:", config["validation_file"])
    print("testing file:", config["testing_file"])
    print("="*100)
