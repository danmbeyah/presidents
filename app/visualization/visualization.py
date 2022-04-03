from flask import Blueprint, render_template
from app.api import fetch_speeches, cluster_speeches
from flask import current_app as app

# Blueprint Configuration
visualization_bp = Blueprint(
    "visualization_bp", __name__, template_folder="templates", static_folder="static"
)

@visualization_bp.route("/visualization", methods=["GET"])
def visualization():
    # Fetch speeches from json file
    speeches = fetch_speeches()

    # Model topics from speeches
    clusters = cluster_speeches(speeches, 10)

    data = {
        "speeches": speeches
    }

    return render_template(
        "visualization.jinja2",
        title = "Presidential Speeches",
        subtitle = "List of few speeches from file.",
        template = "visualization-template",
        data = data,
    )