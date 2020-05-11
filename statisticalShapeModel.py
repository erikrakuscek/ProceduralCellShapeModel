import quilt3
import json

# Fetching cell data #
pkg = quilt3.Package.browse("aics/pipeline_integrated_single_cell", registry="s3://allencell")
cells = pkg["cell_features"]
# cells.fetch("cells")

i = 0
for cell in cells:

    f = open("cells/" + cell)
    cell_data = json.load(f)
    f.close()

    for (k, v) in cell_data.items():
        print(v)

    if i == 10:
        break
    i += 1
