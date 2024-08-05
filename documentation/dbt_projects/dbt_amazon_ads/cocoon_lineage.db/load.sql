COPY column_mapping FROM 'cocoon_lineage.db/column_mapping.csv' (FORMAT 'csv', quote '"', delimiter ',', header 1);
COPY edges FROM 'cocoon_lineage.db/edges.csv' (FORMAT 'csv', quote '"', delimiter ',', header 1);
COPY nodes FROM 'cocoon_lineage.db/nodes.csv' (FORMAT 'csv', quote '"', delimiter ',', header 1);
COPY sql_mapping FROM 'cocoon_lineage.db/sql_mapping.csv' (FORMAT 'csv', quote '"', delimiter ',', header 1);
COPY sql_tags FROM 'cocoon_lineage.db/sql_tags.csv' (FORMAT 'csv', quote '"', delimiter ',', header 1);
COPY table_lineage FROM 'cocoon_lineage.db/table_lineage.csv' (FORMAT 'csv', quote '"', delimiter ',', header 1);