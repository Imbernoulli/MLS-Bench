"""Matching Networks baseline — rigorous codebase edit ops.

Uses bidirectional LSTM to contextualize support features and an attention LSTM
for query features, then classifies via cosine similarity with soft attention
over support labels.

Reference: easyfsl/methods/matching_networks.py

Ops are ordered bottom-to-top so line numbers stay stable.
"""

_FILE = "easy-few-shot-learning/custom_fewshot.py"

_MATCHINGNET = """\
class CustomFewShotMethod(FewShotClassifier):
    \"\"\"Matching Networks (Vinyals et al., 2016).

    Contextualizes support and query features using LSTMs, then classifies
    queries via cosine-similarity-weighted voting over support labels.
    Uses NLLLoss since output is log-probabilities.
    \"\"\"

    # Vinyals et al. 2016 trains MatchingNet with Adam@1e-3 and gradient
    # clipping at 5; SGD@1e-2 (the global default for ProtoNet/RelationNet)
    # destabilizes the bidirectional LSTM and the 25-step query encoder loop,
    # collapsing softmax to uniform output. The framework's training loop
    # honours LR_OVERRIDE to keep this baseline trainable.
    LR_OVERRIDE = 1e-3

    def __init__(self):
        backbone = make_backbone(use_pooling=True)
        super().__init__(backbone=backbone, use_softmax=False)
        self.feature_dimension = FEATURE_DIMENSION

        # Bidirectional LSTM to contextualize support features
        self.support_features_encoder = nn.LSTM(
            input_size=self.feature_dimension,
            hidden_size=self.feature_dimension,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # LSTM cell for attention-based query encoding
        self.query_features_encoding_cell = nn.LSTMCell(
            self.feature_dimension * 2, self.feature_dimension
        )
        self.softmax = nn.Softmax(dim=1)

        self.contextualized_support_features = torch.tensor(())
        self.one_hot_support_labels = torch.tensor(())

    def process_support_set(self, support_images: Tensor, support_labels: Tensor):
        support_features = self.compute_features(support_images)
        self.contextualized_support_features = self._encode_support(support_features)
        self.one_hot_support_labels = F.one_hot(support_labels).float()

    def forward(self, query_images: Tensor) -> Tensor:
        query_features = self.compute_features(query_images)
        contextualized_query_features = self._encode_query(query_features)

        similarity_matrix = self.softmax(
            contextualized_query_features.mm(
                F.normalize(self.contextualized_support_features, dim=1).T
            )
        )
        log_probabilities = (
            similarity_matrix.mm(self.one_hot_support_labels) + 1e-6
        ).log()
        return self.softmax_if_specified(log_probabilities)

    def _encode_support(self, support_features: Tensor) -> Tensor:
        hidden_state = self.support_features_encoder(
            support_features.unsqueeze(0)
        )[0].squeeze(0)
        contextualized = (
            support_features
            + hidden_state[:, : self.feature_dimension]
            + hidden_state[:, self.feature_dimension :]
        )
        return contextualized

    def _encode_query(self, query_features: Tensor) -> Tensor:
        hidden_state = query_features
        cell_state = torch.zeros_like(query_features)

        for _ in range(len(self.contextualized_support_features)):
            attention = self.softmax(
                hidden_state.mm(self.contextualized_support_features.T)
            )
            read_out = attention.mm(self.contextualized_support_features)
            lstm_input = torch.cat((query_features, read_out), 1)
            hidden_state, cell_state = self.query_features_encoding_cell(
                lstm_input, (hidden_state, cell_state)
            )
            hidden_state = hidden_state + query_features

        return hidden_state

    @staticmethod
    def is_transductive() -> bool:
        return False

    def compute_loss(self, scores: Tensor, labels: Tensor) -> Tensor:
        return F.nll_loss(scores, labels)

"""

OPS = [
    {
        "op": "replace",
        "file": _FILE,
        "start_line": 225,
        "end_line": 286,
        "content": _MATCHINGNET,
    },
]
