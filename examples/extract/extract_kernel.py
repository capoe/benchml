import benchml as bml

if __name__ == "__main__":
    bml.log.Connect()
    bml.log.AddArg("models", (list,str), default="^bmol_ecfp_krr$")
    bml.log.AddArg("meta_json", str, default="../../test/test_data/molecular/meta.json")
    args = bml.log.Parse()

    bml.readwrite.configure(use_ase=True)
    models = bml.models.compile_and_filter(["^bmol_ecfp.*$"], args.models)
    datasets = bml.data.DatasetIterator(meta_json=args.meta_json)

    for data in datasets:
        for model in models:
            stream = model.open(data)
            model.precompute(stream)
            for train, test in stream.split(method="random", n_splits=1, train_fraction=100./len(data)):
                model.fit(train) # .fit rather than .map because some descriptors need to be "fitted"
                K = train.resolve('kernel.K')
                print(K)
                print(K.shape)

