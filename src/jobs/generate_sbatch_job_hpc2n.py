import dnnlib


def replace(filedata, string, bystring):
    filedata = filedata.replace(string, bystring)
    return filedata


def prepare_dataset(
    filedata, dataset, num_patients, model, aug, mirror, in_modal, out_modal
):
    if len(in_modal) == 0:
        in_str = ""
    elif len(in_modal) == 1:
        in_str = in_modal[0]
    else:
        in_str = ",".join(in_modal)

    assert len(out_modal) >= 1, "out_modal is empty"
    if len(out_modal) == 1:
        out_str = out_modal[0]
    else:
        out_str = ",".join(out_modal)

    filename = f"scripts/jobs/{dataset:s}_{model:s}_{num_patients:d}_in-{in_str:s}_out-{out_str:s}.sh"

    # Replace the target string
    filedata = replace(filedata, "<dataset>", dataset)
    filedata = replace(filedata, "<num_patients>", str(num_patients))
    filedata = replace(filedata, "<model>", model)
    filedata = replace(filedata, "<aug>", aug)
    filedata = replace(filedata, "<mirror>", str(mirror))
    if in_str == "":
        filedata = replace(filedata, "--in_modal=<in_modal> ", "")
    else:
        filedata = replace(filedata, "<in_modal>", in_str)
    filedata = replace(filedata, "<out_modal>", out_str)

    # Write the file out again
    with open(filename, "w") as file:
        file.write(filedata)

    print(f"sbatch {filename}")


def main():
    # Read in the file
    with open(f"scripts/jobs/template.sh", "r") as file:
        filedata = file.read()

    c = dnnlib.EasyDict()  # Main config dict.
    c.filedata = filedata

    # Prepare the dataset
    for model in ["stylegan2"]:
        for dataset in ["brats20", "ibsr18", "heart", "kits19", "pros", "spleen"]:
        # for dataset in ["spleen"]:
            if dataset == "brats20":
                list_num_patients = [20, 50, 100, 200, 369]
                list_modals = [
                    [[""], [""]],
                    [["truth_label"], ["t1", "t2", "flair", "t1ce", "truth_label"]],
                ]
                mirror = 1
            elif dataset == "ibsr18":
                list_num_patients = [18]
                list_modals = [
                    [[""], [""]],
                    [["truth_label"], ["t1", "truth_label"]],
                ]
                mirror = 1
            elif dataset == "heart":
                list_num_patients = [20]
                list_modals = [
                    [[""], [""]],
                    [["truth_label"], ["img", "truth_label"]],
                ]
                mirror = 0
            elif dataset == "kits19":
                list_num_patients = [20, 50, 100, 203]
                list_modals = [
                    [[""], [""]],
                    [["truth_label"], ["imaging", "truth_label"]],
                ]
                mirror = 0
            elif dataset == "pros":
                list_num_patients = [20, 50, 100, 200, 500]
                list_modals = [
                    [[""], [""]],
                    [["truth_label"], ["ct", "truth_label"]],
                ]
                mirror = 1
            elif dataset == "spleen":
                list_num_patients = [20, 41]
                list_modals = [
                    [[""], [""]],
                    [["truth_label"], ["img", "truth_label"]],
                ]
                mirror = 0
            for num_patients in list_num_patients:
                for in_modal, out_modal in zip(list_modals[0], list_modals[1]):
                    for aug in ["ada"]:
                        c.model = model
                        c.dataset = dataset
                        c.num_patients = num_patients
                        c.in_modal = in_modal
                        c.out_modal = out_modal
                        c.aug = aug
                        c.mirror = mirror
                        prepare_dataset(**c)

    for model in ["condstylegan2", "condstylegan2-v"]:
        for dataset in ["brats20", "ibsr18", "heart", "kits19", "pros", "spleen"]:
        # for dataset in ["kits19"]:
            if dataset == "brats20":
                list_num_patients = [20, 50, 100, 200, 369]
                list_modals = [["truth_label"]], [["t1", "t2", "flair", "t1ce"]]
                mirror = 0
            elif dataset == "ibsr18":
                list_num_patients = [18]
                list_modals = [["truth_label"]], [["t1"]]
                mirror = 0
            elif dataset == "heart":
                list_num_patients = [20]
                list_modals = [["truth_label"]], [["img"]]
                mirror = 0
            elif dataset == "kits19":
                list_num_patients = [20, 50, 100, 203]
                list_modals = [["truth_label"]], [["imaging"]]
                mirror = 0
            elif dataset == "pros":
                list_num_patients = [20, 50, 100, 200, 500]
                list_modals = [["truth_label"]], [["ct"]]
                mirror = 0
            elif dataset == "spleen":
                list_num_patients = [20, 41]
                list_modals = [["truth_label"]], [["img"]]
                mirror = 0
            for num_patients in list_num_patients:
                for in_modal, out_modal in zip(list_modals[0], list_modals[1]):
                    for aug in ["simple"]:
                        c.model = model
                        c.dataset = dataset
                        c.num_patients = num_patients
                        c.in_modal = in_modal
                        c.out_modal = out_modal
                        c.aug = aug
                        c.mirror = mirror
                        prepare_dataset(**c)


if __name__ == "__main__":
    main()
