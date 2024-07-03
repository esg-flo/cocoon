<div align="center">
  <img src="./images/cocoon_logo.png" alt="Cocoon Logo" width="400"/>
</div>

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

😎 Cocoon organizes your data warehouse using LLM agents, preparing it for analysis. Specifically, Cocoon helps you with the tedious steps in data cleaning, data integration, and data modeling. As a result, you can focus on the more intellectual and business-critical parts. Check out the Youtube Deomo 👇:

## Run Scope 3 Category 1 Notebook

The notebook with the name `Scope3_Category1_notebook.ipynb` can be found in the `cocoon_data/notebooks` directory in the repo.

Instructions to setup input variables, run the notebook, and locate the output file can be found in the notebook itself.

## Overview

<br>
<div align="center">
<a href="https://youtu.be/xdmRXs0UnfE" target="_blank">
  <img src="./images/Thumbnail.png" width="600" alt="IMAGE ALT TEXT" style="cursor: pointer;">
</a>
</div>
<br>

- 📚 [Learn more about features](https://cocoon-data-transformation.github.io/page/)
- 💪 Need support? Create an issue or email: zh2408@columbia.edu

😃 Cocoon Data Profiling will be publically available soon!

🖼️ Profile Gallery

| Profile Title                           | Gallery Link                                                                                                                        |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| AQI and Latitude/Longitude of Countries | [View Profile](https://cocoon-data-transformation.github.io/page/profile_gallery/Cocoon_Profile_AQI_and_Lat_Long_of_Countries.html) |
| 2020 Property Sales Data                | [View Profile](https://cocoon-data-transformation.github.io/page/profile_gallery/Cocoon_Profile__2020_property_sales_data.html)     |
| AAC Shelter Cat Outcome                 | [View Profile](https://cocoon-data-transformation.github.io/page/profile_gallery/Cocoon_Profile_aac_shelter_cat_outcome_eng.html)   |
| Books                                   | [View Profile](https://cocoon-data-transformation.github.io/page/profile_gallery/Cocoon_Profile_books.html)                         |
| Cancer                                  | [View Profile](https://cocoon-data-transformation.github.io/page/profile_gallery/Cocoon_Profile_cancer.html)                        |
| Divorces 2000-2015                      | [View Profile](https://cocoon-data-transformation.github.io/page/profile_gallery/Cocoon_Profile_divorces_2000_2015_original.html)   |
| German Credit Data                      | [View Profile](https://cocoon-data-transformation.github.io/page/profile_gallery/Cocoon_Profile_german_credit_data.html)            |
| K-Drama                                 | [View Profile](https://cocoon-data-transformation.github.io/page/profile_gallery/Cocoon_Profile_kdrama.html)                        |
| Patients                                | [View Profile](https://cocoon-data-transformation.github.io/page/profile_gallery/Cocoon_Profile_patients.html)                      |
| Used Car Data                           | [View Profile](https://cocoon-data-transformation.github.io/page/profile_gallery/Cocoon_Profile_used_car_data_new.html)             |

## Get Started

👉 [Try this Google Collab Notebook](https://colab.research.google.com/github/Cocoon-Data-Transformation/cocoon/blob/main/demo/Cocoon_Stage_Demo.ipynb)

Cocoon is available on PyPI:

```bash
pip install cocoon_data
```

To get started, you need to connect to

- LLMs (e.g., GPT-4, Claude-3, Gemini-Ultra, or your local LLMs)
- Data Warehouses (e.g., Snowflake, Big Query, Duckdb...)

```python
from cocoon_data import *

# if you use Open AI GPT-4
openai.api_key  = 'xycabc'

# if you use Snowflake
con = snowflake.connector.connect(...)

query_widget, cocoon_workflow = create_cocoon_workflow(con)

# a helper widget to query your data warehouse
query_widget.display()

# the main panel to interact with Cocoon
cocoon_workflow.start()
```

🎉 You shall see the following on a notebook:

<div align="center">
<kbd><img src="./images/notebook.png" alt=""></kbd>
</div>
If interested, please fill in [waitlist](https://forms.gle/njhNd1NHfh3MvD8V9)
