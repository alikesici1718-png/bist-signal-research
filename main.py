from data_sources.news import collect_news_data
from data_sources.conflict_data import collect_conflict_data
from processors.event_processing import process_events
from processors.clustering import cluster_events
from dashboard.app import run_dashboard

def main():
    news_data = collect_news_data()
    conflict_data = collect_conflict_data()

    events = process_events(news_data, conflict_data)
    clusters = cluster_events(events)

    run_dashboard(clusters)

if __name__ == '__main__':
    main()
