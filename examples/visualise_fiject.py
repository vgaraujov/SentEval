from dataclasses import dataclass
from typing import Dict, List, Tuple
import csv
from pathlib import Path

from fiject import LineGraph, setFijectOutputFolder, ExportMode
from fiject.general import Diagram
import matplotlib.pyplot as plt

# Path setup
PATH_THIS = Path(__file__).resolve()
PATH_ROOT = PATH_THIS.parent.parent
setFijectOutputFolder(PATH_ROOT / "results")

# Visual parameters
Y_LIMS_1 = {
    "Length": (0, 99),
    "WordContent": (0,99),
    "Depth": (15,45),
    "TopConstituents": (10,85),
    "BigramShift": (47, 95),
    "Tense": (47,95),
    "SubjNumber": (47, 90),
    "ObjNumber": (47, 90),
    "OddManOut": (47, 70),
    "CoordinationInversion": (47, 75),

    "MaxCount": (75,84),
    "ArgmaxCount": (39,76),

    "MNIST": (91,99)
}
Y_TICKSPACING_1 = {
    "Length": 20,
    "WordContent": 20,
    "Depth": 10,
    "TopConstituents": 20,
    "BigramShift": 10,
    "Tense": 10,
    "SubjNumber": 10,
    "ObjNumber": 10,
    "OddManOut": 5,
    "CoordinationInversion": 5,

    "MaxCount": 2.5,
    "ArgmaxCount": 10,

    "MNIST": 2
}

TITLE_MAPPING = {  # Optionally do something with this...
    "WordCount": "WC",
    "Depth": "TreeDepth",
    "BigramShift": "BShift",
    "ObjNumber": "ObjNum",
    "OddManOut": "SOMO",
    "CoordinationInversion": "CoordInv"
}

STYLES_1 = {
    "bert": LineGraph.ArgsPerLine(show_points=False, colour="#1F77B4"),
    "pixel": LineGraph.ArgsPerLine(show_points=False, colour="#FF7F0E"),
    "vit-mae": LineGraph.ArgsPerLine(show_points=False, colour="#2CA02C"),
    "baseline": LineGraph.ArgsPerLine(show_points=False, colour="#D62728"),
    "pixel-bigrams": LineGraph.ArgsPerLine(show_points=False, colour="#8C564B")
}

STYLES_2 = {
    "bert": LineGraph.ArgsPerLine(show_points=False, colour="#1F77B4", line_style=":"),
    "vit-mae": LineGraph.ArgsPerLine(show_points=False, colour="#2CA02C", line_style=":"),

    "pixel": LineGraph.ArgsPerLine(show_points=False, colour="#FF7F0E", line_style=":"),
    "pixel-small": LineGraph.ArgsPerLine(show_points=False, colour="#FF7F0E"),
    "pixel-ud": LineGraph.ArgsPerLine(show_points=False, colour="#FF7F0E"),
    "pixel-mnli": LineGraph.ArgsPerLine(show_points=False, colour="#FF7F0E"),

    "baseline": LineGraph.ArgsPerLine(show_points=False, colour="#D62728"),
    "pixel-small-words": LineGraph.ArgsPerLine(show_points=False, colour="#9467BD"),

    "pixel-bigrams": LineGraph.ArgsPerLine(show_points=False, colour="#8C564B", line_style=":"),
    "pixel-small-bigrams": LineGraph.ArgsPerLine(show_points=False, colour="#8C564B"),
    "pixel-bigrams-ud": LineGraph.ArgsPerLine(show_points=False, colour="#8C564B"),
    "pixel-bigrams-mnli": LineGraph.ArgsPerLine(show_points=False, colour="#8C564B"),
}


@dataclass
class FigureParameters:
    name: str

    aspect_ratio: Tuple[float,float]
    scale: float

    line_styles: Dict[str,LineGraph.ArgsPerLine]
    limits: Dict[str, Tuple[int,int]]
    tick_spacings: Dict[str, int]

    vertical_pad: float=0.5
    legend_pad: float=0.05


#######################################################################################################################


def csvToGraphs(csv_path: Path, single_task_column: bool=True) -> Dict[str, LineGraph]:
    print("Loading CSV", csv_path.as_posix())
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)

        if single_task_column:
            graphs: Dict[str,LineGraph] = dict()
            for row in reader:
                task = row.pop("Task")
                model = row.pop("Model").lower()
                layer = int(row.pop("Layer") if "Layer" in row else row.pop(""))
                score = float(row.pop("Accuracy"))

                if task not in graphs:
                    graphs[task] = LineGraph(task)

                graphs[task].add(model, layer, score)
        else:
            graphs: Dict[str,LineGraph] = {k: LineGraph(k) for k in reader.fieldnames if k not in {"", "Model"}}
            for row in reader:
                model = row.pop("Model").lower()
                layer = int(row.pop(""))

                row = {k: float(v) for k,v in row.items()}
                for graph_name, value in row.items():
                    graphs[graph_name].add(model, layer, value)

    return graphs


def fijectGrid(unstructured_graphs: Dict[str, LineGraph], matrix: List[List[str]], config: FigureParameters):
    print("Committing graphs in grid", config.name)

    n_rows = len(matrix)
    n_cols = max(map(len,matrix))

    # Settings for rendering a single tile:
    # aspect_ratio = (3.3,2)
    # FIJECT_DEFAULTS.ASPECT_RATIO_SIZEUP = 0.9  # 0.9 (3.3,2) are my recommended aspect ratio parameters if you want all 12 ticks.
    # FIJECT_DEFAULTS.ASPECT_RATIO_SIZEUP = 0.6  # If you only want 3-6-9-12

    aspect_ratio = (config.aspect_ratio[0]*n_cols,config.aspect_ratio[1]*n_rows)
    fig: plt.Figure = plt.figure(figsize=(config.scale * aspect_ratio[0], config.scale * aspect_ratio[1]))
    axes = fig.subplots(n_rows, n_cols)

    for y, row in enumerate(matrix):
        is_bottom_row = (y == n_rows-1)
        for x, graph_name in enumerate(row):
            ax: plt.Axes = axes[y,x] if n_rows > 1 else axes[x] if n_cols > 1 else axes
            ax.set_title(graph_name, fontsize=8, pad=2)
            graph = unstructured_graphs[graph_name]

            graph.commitWithArgs(
                LineGraph.ArgsGlobal(
                    legend_position="", x_tickspacing=1,
                    y_lims=config.limits.get(graph_name, None), y_tickspacing=config.tick_spacings.get(graph_name, None),
                    curve_linewidth=2, grid_linewidth=0.05,  # Aspect ratio for controlling font size, line widths for controlling thicknesses.
                    x_ticks_hardcoded=[1, 3, 6, 9, 12], x_gridspacing=1,
                    do_spines=False, x_label="Layer"*is_bottom_row, y_label="Accuracy"*(x == 0)
                ),
                LineGraph.ArgsPerLine(show_points=False),
                extra_line_options=config.line_styles,
                existing_figax=(fig,ax), export_mode=ExportMode.RETURN_ONLY
            )

            # Post-processing
            if not is_bottom_row:
                ax.set_xticklabels([])
            else:
                ax.xaxis.labelpad = 1

    # Use only the legend entries of the first subplot.
    legend_handles, legend_labels = (axes[0,0] if n_rows > 1 else axes[0] if n_cols > 1 else axes).get_legend_handles_labels()
    fig.legend(legend_handles, legend_labels,
               loc='upper center', bbox_to_anchor=(0.5, config.legend_pad), ncol=3, frameon=False)
    fig.tight_layout(h_pad=config.vertical_pad, w_pad=0.5)
    Diagram.writeFigure(stem=config.name, suffix=".pdf", figure=fig,
                        overwrite_if_possible=True)


########################################################################################################################


def figure1():
    graphs = csvToGraphs(PATH_ROOT / "results" / "figure_1.csv", single_task_column=False)
    fijectGrid(
        graphs,
        [["Length", "WordContent", "Depth", "TopConstituents", "BigramShift"],
         ["Tense", "SubjNumber", "ObjNumber", "OddManOut", "CoordinationInversion"]],
        FigureParameters(name="figure1", aspect_ratio=(2.6,2), scale=0.65,
                         line_styles=STYLES_1, limits=Y_LIMS_1, tick_spacings=Y_TICKSPACING_1)
    )


def figure3():
    graphs = csvToGraphs(PATH_ROOT / "results" / "figure_3.csv")
    fijectGrid(
        graphs,
        [["MaxCount", "ArgmaxCount"]],
        FigureParameters(name="figure3", aspect_ratio=(2.35, 2), scale=0.9,
                         line_styles=STYLES_1, limits=Y_LIMS_1, tick_spacings=Y_TICKSPACING_1)
    )


def figure4():
    graphs = csvToGraphs(PATH_ROOT / "results" / "figure_4.csv")
    fijectGrid(
        graphs,
        [["MNIST"]],
        FigureParameters(name="figure4", aspect_ratio=(2.5, 2), scale=0.95,
                         line_styles=STYLES_1, limits=Y_LIMS_1, tick_spacings=Y_TICKSPACING_1)
    )


def figure5():
    graphs = csvToGraphs(PATH_ROOT / "results" / "figure_5.csv")
    fijectGrid(
        graphs,
        [["WC", "TreeDepth"],
         ["BShift", "ObjNum"],
         ["SOMO", "CoordInv"]],
        FigureParameters(name="figure5", aspect_ratio=(2.6, 2), scale=0.8,
                         line_styles=STYLES_2, limits=Y_LIMS_1, tick_spacings=Y_TICKSPACING_1,
                         legend_pad=0.025, vertical_pad=0.75)
    )


def figure6():
    graphs = csvToGraphs(PATH_ROOT / "results" / "figure_6.csv")
    fijectGrid(
        graphs,
        [["WC", "TreeDepth"],
         ["BShift", "Tense"],
         ["SOMO", "CoordInv"]],
        FigureParameters(name="figure6", aspect_ratio=(2.6, 2), scale=0.8,
                         line_styles=STYLES_2, limits=dict(), tick_spacings=dict(),
                         legend_pad=0.025, vertical_pad=0.75)
    )


def figure7():
    graphs = csvToGraphs(PATH_ROOT / "results" / "figure_7.csv")
    fijectGrid(
        graphs,
        [["WC", "TreeDepth"],
         ["BShift", "Tense"],
         ["SOMO", "CoordInv"]],
        FigureParameters(name="figure7", aspect_ratio=(2.6, 2), scale=0.8,
                         line_styles=STYLES_2, limits=dict(), tick_spacings=dict(),
                         legend_pad=0.025, vertical_pad=0.75)
    )


def figure8():
    graphs = csvToGraphs(PATH_ROOT / "results" / "figure_8.csv")
    fijectGrid(
        graphs,
        [["WC", "TreeDepth"],
         ["BShift", "Tense"],
         ["SOMO", "CoordInv"]],
        FigureParameters(name="figure8", aspect_ratio=(2.6, 2), scale=0.8,
                         line_styles=STYLES_1, limits=dict(), tick_spacings=dict(),
                         legend_pad=0.025, vertical_pad=0.75)
    )


if __name__ == "__main__":
    figure1()
    figure3()
    figure4()
    figure5()
    figure6()
    figure7()
    figure8()
