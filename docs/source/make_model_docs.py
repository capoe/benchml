from pathlib import Path

import benchml as bml
from benchml.models import collections

log = bml.log
log.setLevel("error")

RST_TOC = """.. toctree::
   :maxdepth: 2
   :caption: Collections of BenchML models:

"""


def make_rst_title(title):
    lines = ["=" * len(title), title, "=" * len(title), "\n"]
    return "\n".join(lines)


create_info = "Creating file {path}."


def main():
    log.Connect()
    log.AddArg("output_dir", str, default="source/models/", help="Provide output directory.")
    log.AddArg(
        "model_content_name", str, default="models.rst", help="Provide name for model content file."
    )
    args = log.Parse()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    collection_list_rst = []
    for group, collection in sorted(collections.items()):
        if str(group) == "null":
            continue
        collection_list_rst.append(f"   {str(group)} <{str(group)}>\n")
        models = collection()
        with open(output_dir / f"{str(group)}.rst", "w") as f:
            print(create_info.format(path=f.name))
            f.write(make_rst_title(f"{str(group)}"))
            for m in models:
                f.write(m.__doc__)
    with open(output_dir / args.model_content_name, "w") as f:
        print(create_info.format(path=f.name))
        f.write(make_rst_title("Models"))
        f.write(RST_TOC)
        f.writelines(collection_list_rst)


if __name__ == "__main__":
    main()
