# unify_utils.py

def unify_columns(df, rename_map: dict):
    """
    Переименовывает колонки согласно rename_map и удаляет старые,
    если имена отличаются.
    """
    df = df.copy()
    for old_col, new_col in rename_map.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
            if old_col != new_col:
                df = df.drop(columns=[old_col])
    return df


def unify_df(df):
    """
    Приводит DataFrame к унифицированному виду:
    - предсказания: pred_relevance
    - ответы: model_response (временно, потом удаляется)
    """
    rename_pred = {
        'agent_pred_relevance': 'pred_relevance',
        'gpt_pred_relevance': 'pred_relevance',
    }

    rename_resp = {
        'agent_response': 'model_response',
        'gpt_response': 'model_response',
    }

    df = unify_columns(df, rename_pred)
    df = unify_columns(df, rename_resp)

    # Удаляем колонку model_response, если есть
    if 'model_response' in df.columns:
        df = df.drop(columns=['model_response'])

    return df
