





with validation_errors as (

    select
        source_relation, account_id, updated_timestamp
    from TEST.PUBLIC_twitter_ads_source.stg_twitter_ads__account_history
    group by source_relation, account_id, updated_timestamp
    having count(*) > 1

)

select *
from validation_errors


