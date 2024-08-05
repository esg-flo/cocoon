





with validation_errors as (

    select
        date_day, app_id, device, platform_version
    from TEST.PUBLIC_apple_store_source.stg_apple_store__crashes_platform_version
    group by date_day, app_id, device, platform_version
    having count(*) > 1

)

select *
from validation_errors

