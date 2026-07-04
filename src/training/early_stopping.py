"""Early stopping baseado na métrica de validação.

Extraído do loop de treino do notebook 06, com a mesma semântica validada
nas runs ``mlp_temporal_v1_*``: melhora **estrita** zera o contador de
paciência; qualquer outra época o incrementa; reduções de learning rate não
interferem no contador.
"""

from dataclasses import dataclass, field


@dataclass
class EarlyStopping:
    """Monitora a métrica de validação (modo *maior é melhor*).

    Uso típico::

        early_stopping = EarlyStopping(patience=10)
        for epoch in range(1, max_epochs + 1):
            metric = validar(model)
            if early_stopping.update(metric, epoch):
                salvar_checkpoint(model)  # nova melhor época
            if early_stopping.should_stop:
                break

    Attributes:
        patience: Nº de épocas sem melhora toleradas antes de parar.
        best_value: Melhor valor observado da métrica.
        best_epoch: Época em que o melhor valor ocorreu.
        epochs_without_improvement: Contador atual de épocas sem melhora.
    """

    patience: int
    best_value: float = field(default=float("-inf"), init=False)
    best_epoch: int = field(default=0, init=False)
    epochs_without_improvement: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError(f"patience deve ser >= 1, recebi {self.patience}.")

    def update(self, value: float, epoch: int) -> bool:
        """Registra a métrica de uma época.

        Args:
            value: Valor da métrica de validação (maior é melhor).
            epoch: Número da época correspondente.

        Returns:
            ``True`` se houve melhora estrita (momento de salvar checkpoint).
        """
        if value > self.best_value:
            self.best_value = value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        """``True`` quando a paciência se esgotou e o treino deve parar."""
        return self.epochs_without_improvement >= self.patience
