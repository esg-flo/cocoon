
    
    

select
    traffic_source_unique_key as unique_field,
    count(*) as n_records

from TEST.PUBLIC_google_play_source.stg_google_play__store_performance_source
where traffic_source_unique_key is not null
group by traffic_source_unique_key
having count(*) > 1

