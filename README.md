Required files:

PostgreSQL dump for re-creating the database of news recommendation system: 10.6084/m9.figshare.27365598

You can also use your own database if the structure matches your database or you edit this Python code. 

Example of use for PostgreSQL:

```
url = sqlalchemy.engine.URL.create(
    drivername="postgresql+psycopg2",
    username=os.environ.get("DB_RECOMMENDER_USER"),
    password=os.environ.get("DB_RECOMMENDER_PASSWORD"),
    host=os.environ.get("DB_RECOMMENDER_HOST"),
    port=os.environ.get("DB_RECOMMENDER_PORT"),
    database=os.environ.get("DB_RECOMMENDER_NAME"),
)
```

set the according enviromental variables with your database credentials (or edit the code).

Nota that the database contains also data not relavant to this particular paper.

For faster processing you can also insert this file into /database folder (although this file should be created automatically if missing): [10.6084/m9.figshare.27365598](https://doi.org/10.6084/m9.figshare.27364983)
