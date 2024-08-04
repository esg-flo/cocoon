





with validation_errors as (

    select
        source_relation, platform, date_day, campaign_id, account_id
    from TEST.PUBLIC_ad_reporting.ad_reporting__campaign_report
    group by source_relation, platform, date_day, campaign_id, account_id
    having count(*) > 1

)

select *
from validation_errors


