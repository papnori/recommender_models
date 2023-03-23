import ast
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def read_data():
    reviews = pd.read_csv("data/reviews.csv")
    products = pd.read_csv("data/products.csv")

    # drop first column
    reviews.drop(columns=reviews.columns[0], axis=1, inplace=True)
    products.drop(columns=products.columns[0], axis=1, inplace=True)

    return reviews, products


def hide_usernames_in(reviews):
    user_map = {name: i for i, name in enumerate(reviews.user_name.unique())}
    reviews.user_name = reviews.user_name.map(user_map)

    return reviews


def process_user_description(reviews):
    user_desc_categories = ["eyes", "hair", "skin tone", "skin"]
    users_desc = []
    for user_description in reviews["user_description"].fillna(""):
        user_desc = {}
        for d in user_description.split(","):
            for category in user_desc_categories:
                if d.endswith(category):
                    user_desc[category] = d.rstrip(" " + category)
        users_desc.append(user_desc)

    return pd.concat([reviews.drop(columns=["user_description"]), pd.DataFrame(users_desc)], axis=1)


def clean_reviews(reviews):
    reviews["stars"] = reviews["stars"].astype(int)
    reviews["recommended"].fillna("Not recommended", inplace=True)

    for col in ['helpful', 'not_helpful']:
        reviews[col] = reviews[col].str.rstrip(')').str.lstrip('(')
        reviews[col] = reviews[col].fillna(0).astype(int)

    return reviews


def one_hot_highlights(products):
    products['highlights'] = products['highlights'].apply(ast.literal_eval)
    mlb = MultiLabelBinarizer()

    res = pd.DataFrame(mlb.fit_transform(products["highlights"]),
                       columns=mlb.classes_,
                       index=products["highlights"].index)

    return pd.concat([products, res], axis=1)


def clean_products(products):
    products["number_of_loves_K"] = products["number_of_loves"].str.rstrip('K').astype(float)
    products.drop(columns=["number_of_loves"], inplace=True)

    products["number_of_reviews"] = [float(num.rstrip('K')) * 1000 if 'K' in num else num for num in
                                     products["number_of_reviews"]]
    products["number_of_reviews"] = products["number_of_reviews"].astype(int)

    products["price_$"] = products["price"].str.lstrip('$').astype(float)
    products.drop(columns=["price"], inplace=True)

    return products


def add_display_name(products):
    products["display_name"] = products["product_name"] + " by " + products["brand_name"]
    return products


def get_data():
    reviews, products = read_data()

    reviews = (reviews.
               pipe(hide_usernames_in).
               pipe(clean_reviews).
               pipe(process_user_description))

    products = (products.
                pipe(clean_products).
                pipe(one_hot_highlights).
                pipe(add_display_name))

    return reviews, products
