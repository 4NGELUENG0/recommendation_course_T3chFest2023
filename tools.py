"""
Tools for the course of recommendation engines.
"""
from os.path import dirname, realpath
from PIL import Image
from typing import Tuple, Dict
from yaml import safe_load
import pandas as pd
import itertools
from sklearn.metrics.pairwise import cosine_similarity

_IMAGES_PATH = "./recursos/datos/images"


def show_article(article_id: str, size: Tuple[int, int] = (200, 200)) -> Image:
    """Function to show an article.

    :param article_id: Id of the article to see.
    :param size: Size of the imagen to show.
    :return: The imagen with the given size.
    """
    three_digits_code = article_id[:3]
    im = Image.open(f"{_IMAGES_PATH}/{three_digits_code}/{article_id}.jpg")
    im = im.resize(size)
    return im


def read_config_data() -> Dict:
    """Read the config.yml file asociated.

    The config.yml file asociated is the one in the same path.

    Returns
    -------
        Dictionary with the configuration of the process.
    """
    base_path = dirname(realpath(__file__))
    config_file_path = f"{base_path}/configuration.yml"
    with open(config_file_path) as conf_file:
        configuration = conf_file.read()
    return safe_load(configuration)


def get_metrics(
    name: str,
    recommendations: pd.DataFrame,
    true_labels: pd.DataFrame,
    articles_df: pd.DataFrame,
    sort_column: str,
    k: int = 5,
    customer_id="cid",
) -> pd.DataFrame:
    """Compute Catalog Coverage, Precision@K, Recall@K, MAP@K and Diversity.

    :param name: Name of the recommender engine.
    :param recommendations: Dataframe with the recommendations per client. Already
        top k recommnedations filtered.
    :param true_labels: Dataframe with the actual items comsuptions per client. This
        dataframe has to contain only the "cid" and the true item.
    :param articles_df: Dataframe with the vectors per article.
    :param sort_colum: Column name to sort recommendations.
    :param k: Number of recommendations.
    :return: Dataframe with the metrics calculated.
    """
    summary = {
        "index": [name],
        "columns": [],
        "data": [[]],
        "index_names": [""],
        "column_names": ["", ""],
    }
    recs = pd.merge(
        left=recommendations,
        right=true_labels,
        how="left",
        on=[customer_id, "article_id"],
    )
    # Catalog coverage
    cat_cov = recommendations["article_id"].nunique() / articles_df.index.nunique()
    summary["columns"].append((f"General", "Catalog coverage (%)"))
    summary["data"][0].extend([round(100 * cat_cov, 4)])

    # Precision@K
    recs["interested"] = recs["interested"].fillna(value=0)
    prec_xcid = recs.groupby(customer_id)["interested"].mean()
    mean_precision_at_k = prec_xcid.mean()
    std_precision_at_k = prec_xcid.std()
    summary["columns"].append((f"Precision@{k}", "mean"))
    summary["columns"].append((f"Precision@{k}", "std"))
    summary["data"][0].extend([mean_precision_at_k, std_precision_at_k])

    # Recall@K
    rec_xcid = recs.groupby(customer_id)["interested"].sum() / 10
    mean_recall_at_k = rec_xcid.mean()
    std_recall_at_k = rec_xcid.std()
    summary["columns"].append((f"Recall@{k}", "mean"))
    summary["columns"].append((f"Recall@{k}", "std"))
    summary["data"][0].extend([mean_recall_at_k, std_recall_at_k])

    # MAP@K
    recs = recs.sort_values(by=[customer_id, sort_column], ascending=[False, False])
    recs["expanding_rolling_sum"] = recs.groupby(customer_id)["interested"].transform(
        lambda x: x.expanding(1).sum()
    )
    recs["expanding_rolling_count"] = recs.groupby(customer_id)["article_id"].transform(
        lambda x: x.expanding(1).count()
    )
    recs["intra_map@k"] = (
        recs["expanding_rolling_sum"] / recs["expanding_rolling_count"]
    )
    recs["intra_map@k"] = recs["intra_map@k"] * recs["interested"]
    map_xcid = recs.groupby(customer_id)["intra_map@k"].mean()
    mean_map_at_k = map_xcid.mean()
    std_map_at_k = map_xcid.std()
    summary["columns"].append((f"MAP@{k}", "mean"))
    summary["columns"].append((f"MAP@{k}", "std"))
    summary["data"][0].extend([mean_map_at_k, std_map_at_k])

    # Diversity
    # TODO: Give to the user the posibility of change the
    # TODO: similarity metric.
    prods_sim = cosine_similarity(
        X=articles_df[articles_df.index.isin(recs["article_id"].unique())],
    )
    sims_mat = pd.DataFrame(
        data=prods_sim,
        columns=recs["article_id"].unique(),
        index=recs["article_id"].unique(),
    )
    sims_df = sims_mat.unstack().reset_index()
    sims_df.columns = ["aid_01", "aid_02", "score"]

    recs = recs.groupby(customer_id)["article_id"].apply(list).reset_index()
    recs["combs"] = recs["article_id"].apply(
        lambda x: list(itertools.combinations(x, 2))
    )
    recs = recs.explode("combs")
    recs["aid_01"], recs["aid_02"] = zip(*recs["combs"])
    recs_diversity = pd.merge(
        left=recs, right=sims_df, how="inner", on=["aid_01", "aid_02"]
    )
    diversity_x_cid = 1 - recs_diversity.groupby(customer_id)["score"].mean()
    mean_diversity = diversity_x_cid.mean()
    std_diversity = diversity_x_cid.std()
    summary["columns"].append((f"Diversity", "mean"))
    summary["columns"].append((f"Diversity", "std"))
    summary["data"][0].extend([mean_diversity, std_diversity])
    return pd.DataFrame.from_dict(summary, orient="tight")
