-- COCOON BLOCK START: PLEASE DO NOT MODIFY THIS BLOCK FOR SELF-MAINTENANCE
-- Generated at 2024-07-22 18:17:55.591589+00:00
WITH 
"E1_rentalInventory_All_renamed" AS (
    -- Rename: Renaming columns
    -- Boro -> Borough
    -- _2010_01 -> date_2010_01
    -- _2010_02 -> date_2010_02
    -- _2010_03 -> date_2010_03
    -- _2010_04 -> date_2010_04
    -- _2010_05 -> date_2010_05
    -- _2010_06 -> date_2010_06
    -- _2010_07 -> date_2010_07
    -- _2010_08 -> date_2010_08
    -- _2010_09 -> date_2010_09
    -- _2010_10 -> date_2010_10
    -- _2010_11 -> date_2010_11
    -- _2010_12 -> date_2010_12
    -- _2011_01 -> date_2011_01
    -- _2011_02 -> date_2011_02
    -- _2011_03 -> date_2011_03
    -- _2011_04 -> date_2011_04
    -- _2011_05 -> date_2011_05
    -- _2011_06 -> date_2011_06
    -- _2011_07 -> date_2011_07
    -- _2011_08 -> date_2011_08
    -- _2011_09 -> date_2011_09
    -- _2011_10 -> date_2011_10
    -- _2011_11 -> date_2011_11
    -- _2011_12 -> date_2011_12
    -- _2012_01 -> date_2012_01
    -- _2012_02 -> date_2012_02
    -- _2012_03 -> date_2012_03
    -- _2012_04 -> date_2012_04
    -- _2012_05 -> date_2012_05
    -- _2012_06 -> date_2012_06
    -- _2012_07 -> date_2012_07
    -- _2012_08 -> date_2012_08
    -- _2012_09 -> date_2012_09
    -- _2012_10 -> date_2012_10
    -- _2012_11 -> date_2012_11
    -- _2012_12 -> date_2012_12
    -- _2013_01 -> date_2013_01
    -- _2013_02 -> date_2013_02
    -- _2013_03 -> date_2013_03
    -- _2013_04 -> date_2013_04
    -- _2013_05 -> date_2013_05
    -- _2013_06 -> date_2013_06
    -- _2013_07 -> date_2013_07
    -- _2013_08 -> date_2013_08
    -- _2013_09 -> date_2013_09
    -- _2013_10 -> date_2013_10
    -- _2013_11 -> date_2013_11
    -- _2013_12 -> date_2013_12
    -- _2014_01 -> date_2014_01
    -- _2014_02 -> date_2014_02
    -- _2014_03 -> date_2014_03
    -- _2014_04 -> date_2014_04
    -- _2014_05 -> date_2014_05
    -- _2014_06 -> date_2014_06
    -- _2014_07 -> date_2014_07
    -- _2014_08 -> date_2014_08
    -- _2014_09 -> date_2014_09
    -- _2014_10 -> date_2014_10
    -- _2014_11 -> date_2014_11
    -- _2014_12 -> date_2014_12
    -- _2015_01 -> date_2015_01
    -- _2015_02 -> date_2015_02
    -- _2015_03 -> date_2015_03
    -- _2015_04 -> date_2015_04
    -- _2015_05 -> date_2015_05
    -- _2015_06 -> date_2015_06
    -- _2015_07 -> date_2015_07
    -- _2015_08 -> date_2015_08
    -- _2015_09 -> date_2015_09
    -- _2015_10 -> date_2015_10
    -- _2015_11 -> date_2015_11
    -- _2015_12 -> date_2015_12
    -- _2016_01 -> date_2016_01
    -- _2016_02 -> date_2016_02
    -- _2016_03 -> date_2016_03
    -- _2016_04 -> date_2016_04
    -- _2016_05 -> date_2016_05
    -- _2016_06 -> date_2016_06
    -- _2016_07 -> date_2016_07
    -- _2016_08 -> date_2016_08
    -- _2016_09 -> date_2016_09
    -- _2016_10 -> date_2016_10
    -- _2016_11 -> date_2016_11
    -- _2016_12 -> date_2016_12
    -- _2017_01 -> date_2017_01
    -- _2017_02 -> date_2017_02
    -- _2017_03 -> date_2017_03
    -- _2017_04 -> date_2017_04
    -- _2017_05 -> date_2017_05
    -- _2017_06 -> date_2017_06
    -- _2017_07 -> date_2017_07
    -- _2017_08 -> date_2017_08
    -- _2017_09 -> date_2017_09
    -- _2017_10 -> date_2017_10
    -- _2017_11 -> date_2017_11
    -- _2017_12 -> date_2017_12
    -- _2018_01 -> date_2018_01
    -- _2018_02 -> date_2018_02
    -- _2018_03 -> date_2018_03
    -- _2018_04 -> date_2018_04
    -- _2018_05 -> date_2018_05
    -- _2018_06 -> date_2018_06
    -- _2018_07 -> date_2018_07
    -- _2018_08 -> date_2018_08
    -- _2018_09 -> date_2018_09
    SELECT 
        "Area",
        "Boro" AS "Borough",
        "AreaType",
        "_2010_01" AS "date_2010_01",
        "_2010_02" AS "date_2010_02",
        "_2010_03" AS "date_2010_03",
        "_2010_04" AS "date_2010_04",
        "_2010_05" AS "date_2010_05",
        "_2010_06" AS "date_2010_06",
        "_2010_07" AS "date_2010_07",
        "_2010_08" AS "date_2010_08",
        "_2010_09" AS "date_2010_09",
        "_2010_10" AS "date_2010_10",
        "_2010_11" AS "date_2010_11",
        "_2010_12" AS "date_2010_12",
        "_2011_01" AS "date_2011_01",
        "_2011_02" AS "date_2011_02",
        "_2011_03" AS "date_2011_03",
        "_2011_04" AS "date_2011_04",
        "_2011_05" AS "date_2011_05",
        "_2011_06" AS "date_2011_06",
        "_2011_07" AS "date_2011_07",
        "_2011_08" AS "date_2011_08",
        "_2011_09" AS "date_2011_09",
        "_2011_10" AS "date_2011_10",
        "_2011_11" AS "date_2011_11",
        "_2011_12" AS "date_2011_12",
        "_2012_01" AS "date_2012_01",
        "_2012_02" AS "date_2012_02",
        "_2012_03" AS "date_2012_03",
        "_2012_04" AS "date_2012_04",
        "_2012_05" AS "date_2012_05",
        "_2012_06" AS "date_2012_06",
        "_2012_07" AS "date_2012_07",
        "_2012_08" AS "date_2012_08",
        "_2012_09" AS "date_2012_09",
        "_2012_10" AS "date_2012_10",
        "_2012_11" AS "date_2012_11",
        "_2012_12" AS "date_2012_12",
        "_2013_01" AS "date_2013_01",
        "_2013_02" AS "date_2013_02",
        "_2013_03" AS "date_2013_03",
        "_2013_04" AS "date_2013_04",
        "_2013_05" AS "date_2013_05",
        "_2013_06" AS "date_2013_06",
        "_2013_07" AS "date_2013_07",
        "_2013_08" AS "date_2013_08",
        "_2013_09" AS "date_2013_09",
        "_2013_10" AS "date_2013_10",
        "_2013_11" AS "date_2013_11",
        "_2013_12" AS "date_2013_12",
        "_2014_01" AS "date_2014_01",
        "_2014_02" AS "date_2014_02",
        "_2014_03" AS "date_2014_03",
        "_2014_04" AS "date_2014_04",
        "_2014_05" AS "date_2014_05",
        "_2014_06" AS "date_2014_06",
        "_2014_07" AS "date_2014_07",
        "_2014_08" AS "date_2014_08",
        "_2014_09" AS "date_2014_09",
        "_2014_10" AS "date_2014_10",
        "_2014_11" AS "date_2014_11",
        "_2014_12" AS "date_2014_12",
        "_2015_01" AS "date_2015_01",
        "_2015_02" AS "date_2015_02",
        "_2015_03" AS "date_2015_03",
        "_2015_04" AS "date_2015_04",
        "_2015_05" AS "date_2015_05",
        "_2015_06" AS "date_2015_06",
        "_2015_07" AS "date_2015_07",
        "_2015_08" AS "date_2015_08",
        "_2015_09" AS "date_2015_09",
        "_2015_10" AS "date_2015_10",
        "_2015_11" AS "date_2015_11",
        "_2015_12" AS "date_2015_12",
        "_2016_01" AS "date_2016_01",
        "_2016_02" AS "date_2016_02",
        "_2016_03" AS "date_2016_03",
        "_2016_04" AS "date_2016_04",
        "_2016_05" AS "date_2016_05",
        "_2016_06" AS "date_2016_06",
        "_2016_07" AS "date_2016_07",
        "_2016_08" AS "date_2016_08",
        "_2016_09" AS "date_2016_09",
        "_2016_10" AS "date_2016_10",
        "_2016_11" AS "date_2016_11",
        "_2016_12" AS "date_2016_12",
        "_2017_01" AS "date_2017_01",
        "_2017_02" AS "date_2017_02",
        "_2017_03" AS "date_2017_03",
        "_2017_04" AS "date_2017_04",
        "_2017_05" AS "date_2017_05",
        "_2017_06" AS "date_2017_06",
        "_2017_07" AS "date_2017_07",
        "_2017_08" AS "date_2017_08",
        "_2017_09" AS "date_2017_09",
        "_2017_10" AS "date_2017_10",
        "_2017_11" AS "date_2017_11",
        "_2017_12" AS "date_2017_12",
        "_2018_01" AS "date_2018_01",
        "_2018_02" AS "date_2018_02",
        "_2018_03" AS "date_2018_03",
        "_2018_04" AS "date_2018_04",
        "_2018_05" AS "date_2018_05",
        "_2018_06" AS "date_2018_06",
        "_2018_07" AS "date_2018_07",
        "_2018_08" AS "date_2018_08",
        "_2018_09" AS "date_2018_09"
    FROM "memory"."main"."E1_rentalInventory_All"
)

-- COCOON BLOCK END
SELECT *
FROM "E1_rentalInventory_All_renamed"