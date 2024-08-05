

with hourly as (
    
    select *
    from TEST.PUBLIC_stg_tiktok_ads.stg_tiktok_ads__campaign_report_hourly
), 

campaigns as (

    select *
    from TEST.PUBLIC_stg_tiktok_ads.stg_tiktok_ads__campaign_history
    where is_most_recent_record
), 

advertiser as (

    select *
    from TEST.PUBLIC_stg_tiktok_ads.stg_tiktok_ads__advertiser
), 

aggregated as (

    select
        hourly.source_relation,
        cast(hourly.stat_time_hour as date) as date_day,
        advertiser.advertiser_id,
        advertiser.advertiser_name,
        hourly.campaign_id,
        campaigns.campaign_name,
        advertiser.currency,
        sum(hourly.impressions) as impressions,
        sum(hourly.clicks) as clicks,
        sum(hourly.spend) as spend,
        sum(hourly.reach) as reach,
        sum(hourly.conversion) as conversion,
        sum(hourly.likes) as likes,
        sum(hourly.comments) as comments,
        sum(hourly.shares) as shares,
        sum(hourly.profile_visits) as profile_visits,
        sum(hourly.follows) as follows,
        sum(hourly.video_watched_2_s) as video_watched_2_s,
        sum(hourly.video_watched_6_s) as video_watched_6_s,
        sum(hourly.video_views_p_25) as video_views_p_25,
        sum(hourly.video_views_p_50) as video_views_p_50, 
        sum(hourly.video_views_p_75) as video_views_p_75,
        sum(hourly.spend)/nullif(sum(hourly.clicks),0) as daily_cpc,
        (sum(hourly.spend)/nullif(sum(hourly.impressions),0))*1000 as daily_cpm,
        (sum(hourly.clicks)/nullif(sum(hourly.impressions),0))*100 as daily_ctr

        




    
    from hourly
    left join campaigns
        on hourly.campaign_id = campaigns.campaign_id
        and hourly.source_relation = campaigns.source_relation
    left join advertiser
        on campaigns.advertiser_id = advertiser.advertiser_id
        and campaigns.source_relation = advertiser.source_relation
    group by 1,2,3,4,5,6,7

)

select *
from aggregated