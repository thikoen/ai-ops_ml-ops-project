"""Hopsworks connection, Feature Group and Feature View definitions."""

from .settings import FEATURES, GROUP, TARGET, VIEW


def connect(settings):
    import hopsworks

    if not settings.project or not settings.api_key:
        raise ValueError("Set HOPSWORKS_PROJECT and HOPSWORKS_API_KEY in .env")
    return hopsworks.login(
        host=settings.host,
        project=settings.project,
        api_key_value=settings.api_key,
        engine="python",
        hostname_verification=True,
        cert_folder=str((settings.work / "hopsworks-certs").resolve()),
    )


def feature_group(project, version):
    return project.get_feature_store().get_or_create_feature_group(
        name=GROUP,
        version=version,
        primary_key=["trip_id"],
        event_time="event_time",
        time_travel_format="DELTA",
        online_enabled=False,
        statistics_config=False,
        description="ICE Hamburg Hbf to Berlin Hauptbahnhof; historical replay; label arrival delay >=5min",
    )


def feature_view(project, version):
    fs = project.get_feature_store()
    group = fs.get_feature_group(GROUP, version=version)
    return fs.get_or_create_feature_view(
        name=VIEW,
        version=version,
        labels=[TARGET],
        query=group.select(["trip_id", "event_time", "label_available_at", *FEATURES, TARGET]),
        training_helper_columns=["label_available_at"],
        description="Seven-day historical delay, current departure delay and calendar features",
    )
