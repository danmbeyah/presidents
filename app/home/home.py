from flask import Blueprint
from flask import current_app as app
from flask import render_template

# Blueprint Configuration
home_bp = Blueprint(
    "home_bp", __name__, template_folder="templates", static_folder="static"
)


@home_bp.route("/", methods=["GET"])
def home():
    return render_template(
        "index.jinja2",
        title="Graph Network Visualization of Presidential Speeches",
        subtitle="",
        template="home-template",
    )


@home_bp.route("/team", methods=["GET"])
def team():
    return render_template(
        "team.jinja2",
        title="Team Members",
        subtitle="This was a team project implemented as a fulfillment towards a Msc. Computer Science degree.",
        template="home-template page",
    )