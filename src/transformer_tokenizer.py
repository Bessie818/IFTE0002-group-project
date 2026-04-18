import numpy as np
import pandas as pd


class TokenizerArtifacts:
    def __init__(self, vocab, numerical_bin_edges):
        self.vocab = vocab
        self.numerical_bin_edges = numerical_bin_edges


class CreditCardTokenizer:
    def __init__(
        self,
        categorical_features,
        numerical_features,
        feature_order,
        num_bins=8,
        pad_token="[PAD]",
        cls_token="[CLS]",
        unk_token="[UNK]",
        missing_token_suffix="MISSING",
    ):
        self.categorical_features = list(categorical_features)
        self.numerical_features = list(numerical_features)
        self.feature_order = list(feature_order)
        self.sequence_length = 1 + len(self.feature_order)
        self.num_bins = num_bins
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.unk_token = unk_token
        self.missing_token_suffix = missing_token_suffix

        self.vocab = {}
        self.inverse_vocab = {}
        self.numerical_bin_edges = {}
        self.vocab_size = 0
        self.is_fitted = False

    def fit(self, df):
        vocab_tokens = [self.pad_token, self.cls_token, self.unk_token]

        for feature in self.categorical_features:
            observed_values = (
                df[feature]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )
            vocab_tokens.extend(f"{feature}={value}" for value in observed_values)
            vocab_tokens.append(f"{feature}={self.missing_token_suffix}")

        for feature in self.numerical_features:
            series = pd.to_numeric(df[feature], errors="coerce")
            quantiles = np.linspace(0.0, 1.0, self.num_bins + 1)
            raw_edges = np.unique(np.quantile(series.dropna(), quantiles))
            if raw_edges.size < 2:
                raw_edges = np.array([series.min(), series.max()], dtype=float)
            if raw_edges.size < 2 or np.isnan(raw_edges).all():
                raw_edges = np.array([0.0, 1.0], dtype=float)

            edges = raw_edges.astype(float)
            edges[0] = -np.inf
            edges[-1] = np.inf
            self.numerical_bin_edges[feature] = edges

            for bin_index in range(len(edges) - 1):
                vocab_tokens.append(f"{feature}_bin_{bin_index}")
            vocab_tokens.append(f"{feature}_{self.missing_token_suffix}")

        deduplicated_tokens = []
        seen = set()
        for token in vocab_tokens:
            if token not in seen:
                deduplicated_tokens.append(token)
                seen.add(token)

        self.vocab = {token: index for index, token in enumerate(deduplicated_tokens)}
        self.inverse_vocab = {index: token for token, index in self.vocab.items()}
        self.vocab_size = len(self.vocab)
        self.is_fitted = True
        return self

    def transform_to_tokens(self, df):
        self._require_fitted()
        records = []
        for _, row in df.iterrows():
            record_tokens = [self.cls_token]
            for feature in self.feature_order:
                if feature in self.categorical_features:
                    record_tokens.append(self._categorical_token(feature, row[feature]))
                elif feature in self.numerical_features:
                    record_tokens.append(self._numerical_token(feature, row[feature]))
                else:
                    raise KeyError(f"Feature {feature} is not registered in tokenizer.")
            records.append(record_tokens)
        return records

    def encode(self, df):
        token_sequences = self.transform_to_tokens(df)
        encoded = np.full(
            (len(token_sequences), self.sequence_length),
            fill_value=self.vocab[self.pad_token],
            dtype=np.int64,
        )
        unk_id = self.vocab[self.unk_token]

        for row_index, tokens in enumerate(token_sequences):
            encoded[row_index, : len(tokens)] = [
                self.vocab.get(token, unk_id) for token in tokens
            ]
        return encoded

    def decode(self, ids):
        self._require_fitted()
        return [self.inverse_vocab[int(index)] for index in ids]

    def export_artifacts(self):
        self._require_fitted()
        return TokenizerArtifacts(
            vocab=self.vocab,
            numerical_bin_edges=self.numerical_bin_edges,
        )

    def _categorical_token(self, feature, value):
        if pd.isna(value):
            return f"{feature}={self.missing_token_suffix}"
        return f"{feature}={str(value)}"

    def _numerical_token(self, feature, value):
        if pd.isna(value):
            return f"{feature}_{self.missing_token_suffix}"

        numeric_value = float(value)
        edges = self.numerical_bin_edges[feature]
        bin_index = int(np.digitize(numeric_value, edges[1:-1], right=False))
        return f"{feature}_bin_{bin_index}"

    def _require_fitted(self):
        if not self.is_fitted:
            raise RuntimeError("Tokenizer must be fitted before use.")
