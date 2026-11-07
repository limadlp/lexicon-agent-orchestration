"""Public domain benchmark dataset: The Wonderful Wizard of Oz (L. Frank Baum, 1900)."""

from __future__ import annotations

from lexicon.models import CharacterEntry, ChapterDocument, TermEntry


def build_oz_dataset() -> tuple[list[ChapterDocument], dict[int, str], list[TermEntry], list[CharacterEntry], list[TermEntry]]:
    """Build the 3-chapter evaluation corpus, seed glossary, and ground-truth lexicon.

    Returns:
        (chapters, reference_translations, seed_terms, seed_characters, ground_truth_terms)
    """
    chapters = [
        ChapterDocument(
            number=1,
            filename="001_The_Council_with_the_Munchkins.md",
            title="The Council with the Munchkins",
            content=(
                "# Chapter 1: The Council with the Munchkins\n\n"
                "The counselor of the North greeted Dorothy near the yellow brick road. "
                "The counselor was cautious after observing the winged monkeys scatter across "
                "the eastern hills, warning that only the great wizard could help.\n\n"
                "Before departing, the young traveler examined the ancient golden cap. "
                "The traveler was exhausted from the cyclone, yet adjusted the emerald spectacles "
                "stored inside the basket."
            ),
        ),
        ChapterDocument(
            number=2,
            filename="002_The_Gate_of_the_Emerald_City.md",
            title="The Gate of the Emerald City",
            content=(
                "# Chapter 2: The Gate of the Emerald City\n\n"
                "At the palace threshold, Dorothy held the mystic golden cap tightly. "
                "The young traveler was determined to speak with the wizard before the winged monkeys "
                "returned from the western forest.\n\n"
                "The ruler of the South sent a message across the court. The ruler was patient, "
                "reminding every citizen to wear the locked emerald spectacles inside the citadel."
            ),
        ),
        ChapterDocument(
            number=3,
            filename="003_The_Winged_Monkeys_Summons.md",
            title="The Winged Monkeys Summons",
            content=(
                "# Chapter 3: The Winged Monkeys Summons\n\n"
                "The ruler of the South inspected the inscriptions carved inside the golden cap. "
                "The ruler was confident that a single wizard could command the winged monkeys "
                "three times without breaking the emerald spectacles.\n\n"
                "The counselor of the North nodded in agreement. The counselor was grateful "
                "that the young traveler was safe."
            ),
        ),
    ]

    references = {
        1: (
            "# Capítulo 1: O Conselho com os Munchkins\n\n"
            "A conselheira do Norte cumprimentou Dorothy perto da estrada de tijolos amarelos. "
            "A conselheira foi cautelosa após observar os macacos alados se espalharem pelas "
            "colinas do leste, alertando que apenas o grande mágico poderia ajudar.\n\n"
            "Antes de partir, a jovem viajante examinou o antigo gorrete de ouro. "
            "A viajante estava exausta do ciclone, mas ajustou os óculos de esmeralda "
            "guardados dentro da cesta."
        ),
        2: (
            "# Capítulo 2: O Portão da Cidade das Esmeraldas\n\n"
            "No limiar do palácio, Dorothy segurou o místico gorrete de ouro com firmeza. "
            "A jovem viajante estava determinada a falar com o mágico antes que os macacos alados "
            "retornassem da floresta ocidental.\n\n"
            "A governante do Sul enviou uma mensagem pela corte. A governante foi paciente, "
            "lembrando cada cidadão de usar os óculos de esmeralda trancados dentro da cidadela."
        ),
        3: (
            "# Capítulo 3: O Chamado dos Macacos Alados\n\n"
            "A governante do Sul inspecionou as inscrições entalhadas dentro do gorrete de ouro. "
            "A governante estava confiante de que um único mágico poderia comandar os macacos alados "
            "três vezes sem quebrar os óculos de esmeralda.\n\n"
            "A conselheira do Norte assentiu em concordância. A conselheira ficou grata "
            "por a jovem viajante estar segura."
        ),
    }

    seed_terms = [
        TermEntry(
            source="wizard",
            target="mágico",
            forbidden_synonyms=("bruxo", "mago", "sábio", "feiticeiro"),
            notes="Traduzir wizard canonicamente como mágico (O Mágico de Oz; bruxo/mago/sábio são drift).",
        ),
        TermEntry(
            source="winged monkeys",
            target="macacos alados",
            aliases=("winged monkey",),
            forbidden_synonyms=("símios voadores", "macacos com asas"),
            notes="Criaturas encantadas sob comando do gorrete.",
        ),
        TermEntry(
            source="emerald spectacles",
            target="óculos de esmeralda",
            forbidden_synonyms=("lentes verdes", "visores de esmeralda"),
            notes="Óculos protetores obrigatórios na Cidade das Esmeraldas.",
        ),
    ]

    seed_characters = [
        CharacterEntry(
            name="Glinda",
            target_name="Glinda",
            gender="F",
            english_referents=("the ruler of the south", "the ruler"),
            expected_markers_pt=("a governante do sul", "a governante"),
            forbidden_markers_pt=("o governante do sul", "o governante foi paciente"),
            notes="Governante do Sul; gênero feminino.",
        ),
        CharacterEntry(
            name="Tattypoo",
            target_name="Tattypoo",
            gender="F",
            english_referents=("the counselor of the north", "the counselor"),
            expected_markers_pt=("a conselheira do norte", "a conselheira"),
            forbidden_markers_pt=("o conselheiro do norte", "o conselheiro foi cauteloso"),
            notes="Bruxa Boa do Norte; conselheira feminina.",
        ),
        CharacterEntry(
            name="Dorothy",
            target_name="Dorothy",
            gender="F",
            english_referents=("the young traveler", "the traveler"),
            expected_markers_pt=("a jovem viajante", "a viajante estava exausta"),
            forbidden_markers_pt=("o jovem viajante", "o viajante estava exausto"),
            notes="Viajante protagonista; gênero feminino.",
        ),
    ]

    ground_truth_terms = [
        *seed_terms,
        TermEntry(
            source="golden cap",
            target="gorrete de ouro",
            forbidden_synonyms=("chapéu dourado", "coroa áurea"),
            notes="Artefato mágico eleito pelo capítulo 1.",
        ),
    ]

    return chapters, references, seed_terms, seed_characters, ground_truth_terms
