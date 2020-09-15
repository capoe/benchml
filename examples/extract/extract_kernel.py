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
            model.fit(stream)
            K = stream.resolve('kernel.K')
            print(K)
            input('...')

