import benchml as bml

if __name__ == "__main__":
    bml.log.Connect()
    bml.log.AddArg("models", (list, str), default=["^bmol_soap.*_krr$"])
    bml.log.AddArg("meta_json", str, default="../../test/test_data/molecular/meta.json")
    args = bml.log.Parse()

    bml.readwrite.configure(use_ase=True)
    bml.splits.synchronize(seed=0)

    models = list(bml.models.compile_and_filter(["^bmol_.*$"], args.models))
    datasets = bml.data.DatasetIterator(meta_json=args.meta_json)
    for data in datasets:
        for model in models:
            stream = model.open(data)
            for train, test in stream.split(
                method="random", n_splits=1, train_fraction=100.0 / len(data)
            ):
                model.fit(
                    train, endpoint="kernel"
                )  # .fit rather than .map because some descriptors need to be "fitted"
                K = train.resolve("kernel.K")
                X = train.resolve("descriptor.X")
                print(K)
                print(K.shape)
                print(X)
                print(X.shape)
                input("< voila, the kernel for model %s" % model.tag)
