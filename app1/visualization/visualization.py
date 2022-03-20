"""Routes for logged-in profile."""
from faker import Faker
from flask import Blueprint, render_template
from app1.api import fetch_speeches
from flask import current_app as app

fake = Faker()

# Blueprint Configuration
visualization_bp = Blueprint(
    "visualization_bp", __name__, template_folder="templates", static_folder="static"
)

@visualization_bp.route("/visualization", methods=["GET"])
def visualization():
    speeches = fetch_speeches()
    return render_template(
        "visualization.jinja2",
        title="Presidential Speeches",
        subtitle="List of few speeches from file.",
        template="visualization-template",
        speeches=speeches,
    )