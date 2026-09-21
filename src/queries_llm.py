import json

# Input-output examples to guide the LLM response format.
FEW_SHOTS = {
    "default_tabular": [
        {
            "user": {
                "query": "Fuentes de agua potable en el municipio",
                "result": {
                    "title": "Fuentes de agua potable de Madrid",
                    "description": "Inventario municipal de fuentes con agua apta para el consumo humano",
                    "header": "\"ID\";\"NOMBRE\";\"DISTRITO\";\"DIRECCION\";\"TIPO\";\"ESTADO\";\"LATITUD\";\"LONGITUD\"",
                    "content": "\"001\";\"Fuente Plaza Mayor\";\"CENTRO\";\"Plaza Mayor, s/n\";\"POTABLE\";\"ACTIVA\";\"40.415\";\"-3.707\"",
                },
            },
            "assistant": {
                "main_resource": "fuentes de agua potable",
                "scope": "municipal",
                "supporting_field": "content",
                "quote": "\"Fuente Plaza Mayor\";\"CENTRO\";\"Plaza Mayor, s/n\";\"POTABLE\";\"ACTIVA\"",
                "reason": "direct_match",
            },
        },
        {
            "user": {
                "query": "Centros de dia para personas mayores",
                "result": {
                    "title": "Centros de dia municipales para mayores dependientes",
                    "description": "Instalaciones publicas de atencion diurna",
                    "header": "\"NOMBRE\";\"TIPO\";\"DISTRITO\";\"DIRECCION\";\"PLAZAS\";\"LATITUD\";\"LONGITUD\"",
                    "content": "\"Centro de Dia Vista Alegre\";\"MUNICIPAL\";\"CARABANCHEL\";\"C/ General Ricardos, 177\";\"60\";\"40.385\";\"-3.736\"",
                },
            },
            "assistant": {
                "main_resource": "centros de dia para personas mayores",
                "scope": None,
                "supporting_field": "title",
                "quote": "Centros de dia municipales para mayores dependientes",
                "reason": "subtype",
            },
        },
        {
            "user": {
                "query": "Piscinas publicas cubiertas por barrio",
                "result": {
                    "title": "Centros deportivos municipales",
                    "description": "Relacion de instalaciones deportivas del Ayuntamiento de Madrid",
                    "header": "\"NOMBRE\";\"TIPO\";\"DISTRITO\";\"DIRECCION\";\"SUPERFICIE\"",
                    "content": "\"Centro Deportivo Moratalaz\";\"POLIDEPORTIVO\";\"MORATALAZ\";\"Av. Moratalaz, 12\";\"4200\"",
                },
            },
            "assistant": {
                "main_resource": "piscinas cubiertas",
                "scope": "barrio",
                "supporting_field": None,
                "quote": None,
                "reason": "generic",
            },
        },
        {
            "user": {
                "query": "Calidad del aire",
                "result": {
                    "title": "Niveles de ruido por distrito",
                    "description": "Mediciones acusticas en distintos puntos de la ciudad",
                    "header": "\"DISTRITO\";\"NIVEL SONORO MEDIO (dB)\";\"FECHA\"",
                    "content": "\"CENTRO\";\"68\";\"2024-05-13\"",
                },
            },
            "assistant": {
                "main_resource": "calidad del aire",
                "scope": None,
                "supporting_field": None,
                "quote": None,
                "reason": "unrelated",
            },
        },
        {
            "user": {
                "query": "Subvenciones para rehabilitacion de vivienda en madrid",
                "result": {
                    "title": "Ayudas y subvenciones para vivienda",
                    "description": "Convocatoria anual de subvenciones para la rehabilitacion de viviendas en el municipio de Madrid.",
                    "header": "\"ANIO\";\"LINEA\";\"IMPORTE\";\"MUNICIPIO\"",
                    "content": "\"2025\";\"Rehabilitacion residencial\";\"2500000\";\"Madrid\"",
                },
            },
            "assistant": {
                "main_resource": "subvenciones para rehabilitacion de vivienda",
                "scope": "municipal",
                "supporting_field": "description",
                "quote": "subvenciones para la rehabilitacion de viviendas en el municipio de Madrid",
                "reason": "direct_match",
            },
        },
        {
            "user": {
                "query": "Farmacias de guardia por distrito",
                "result": {
                    "title": "Directorio de farmacias",
                    "description": "Listado actualizado de farmacias y turnos en la ciudad.",
                    "header": "\"DISTRITO\";\"FARMACIA DE GUARDIA\";\"DIRECCION\";\"TELEFONO\"",
                    "content": "\"CENTRO\";\"Farmacia Sol\";\"Calle Arenal 12\";\"910000001\"",
                },
            },
            "assistant": {
                "main_resource": "farmacias de guardia",
                "scope": "distrito",
                "supporting_field": "header",
                "quote": "\"DISTRITO\";\"FARMACIA DE GUARDIA\";\"DIRECCION\";\"TELEFONO\"",
                "reason": "direct_match",
            },
        },
        {
            "user": {
                "query": "Centros de salud mental infantil",
                "result": {
                    "title": "Servicios de salud",
                    "description": "Informacion general de recursos sanitarios.",
                    "header": "\"CODIGO\";\"NOMBRE\";\"TIPO\"",
                    "content": "\"01\";\"Centro A\";\"Sanitario\"",
                },
            },
            "assistant": {
                "main_resource": "centros de salud mental infantil",
                "scope": None,
                "supporting_field": None,
                "quote": None,
                "reason": "insufficient_evidence",
            },
        },
        {
            "user": {
                "query": "Becas de comedor escolar",
                "result": {
                    "title": "Ayudas educativas",
                    "description": "Dataset con distintos programas de ayuda para estudiantes.",
                    "header": "\"PROGRAMA\";\"BENEFICIARIOS\";\"ANIO\"",
                    "content": "\"Transporte escolar\";\"12000\";\"2024\"",
                },
            },
            "assistant": {
                "main_resource": "becas de comedor escolar",
                "scope": None,
                "supporting_field": None,
                "quote": None,
                "reason": "generic",
            },
        },
        {
            "user": {
                "query": "Calidad del agua potable",
                "result": {
                    "title": "Calidad del aire urbano",
                    "description": "Mediciones de NO2, PM10 y ozono por estacion.",
                    "header": "\"ESTACION\";\"NO2\";\"PM10\";\"FECHA\"",
                    "content": "\"Plaza Espana\";\"42\";\"18\";\"2025-02-10\"",
                },
            },
            "assistant": {
                "main_resource": "calidad del agua potable",
                "scope": None,
                "supporting_field": None,
                "quote": None,
                "reason": "unrelated",
            },
        },
        {
            "user": {
                "query": "Puntos de recarga rapida para vehiculos electricos",
                "result": {
                    "title": "Infraestructura de recarga de vehiculos electricos",
                    "description": "Ubicacion de puntos de recarga rapida y semirrapida en via publica.",
                    "header": "\"NOMBRE\";\"TIPO_CARGA\";\"LAT\";\"LON\"",
                    "content": "\"Estacion Norte\";\"semirrapida\";\"40.42\";\"-3.70\"",
                },
            },
            "assistant": {
                "main_resource": "puntos de recarga rapida para vehiculos electricos",
                "scope": None,
                "supporting_field": "description",
                "quote": "puntos de recarga rapida y semirrapida",
                "reason": "subtype",
            },
        },
        {
            "user": {
                "query": "Centros culturales en el distrito centro",
                "result": {
                    "title": "Centros culturales municipales de todos los distritos",
                    "description": "Relacion completa de centros culturales de la ciudad.",
                    "header": "\"NOMBRE\";\"DISTRITO\";\"DIRECCION\"",
                    "content": "\"Centro Cultural X\";\"SALAMANCA\";\"Calle Velazquez 10\"",
                },
            },
            "assistant": {
                "main_resource": "centros culturales",
                "scope": "distrito centro",
                "supporting_field": None,
                "quote": None,
                "reason": "generic",
            },
        },
        {
            "user": {
                "query": "Accidentes de trafico con bicicletas",
                "result": {
                    "title": "Accidentes de trafico",
                    "description": "Registro de siniestros viales con detalle de vehiculos implicados.",
                    "header": "\"FECHA\";\"DISTRITO\";\"TIPO_VEHICULO\";\"GRAVEDAD\"",
                    "content": "\"2025-01-11\";\"CENTRO\";\"bicicleta\";\"leve\"",
                },
            },
            "assistant": {
                "main_resource": "accidentes de trafico con bicicletas",
                "scope": None,
                "supporting_field": "content",
                "quote": "\"2025-01-11\";\"CENTRO\";\"bicicleta\";\"leve\"",
                "reason": "subtype",
            },
        },
    ],
}

# Expected output format from the model.
OUTPUT_FORMAT = {
    "default_tabular":
        """
{
    "main_resource": "<string>",
    "scope": "<string or null>",
    "supporting_field": "<title | description | header | content | null>",
    "quote": "<exact substring or null>",
    "reason": "<direct_match | subtype | generic | unrelated | insufficient_evidence>"
}
        """,
}

# Evaluation prompt templates.
EVALUATION_INSTRUCTIONS = {
    "default_tabular":
        """
You are a relevance judge for dataset retrieval.

Your task is to classify the relevance relationship between one retrieved result and one user query.
Return exactly one JSON object and nothing else.

# Task
Given:
- one query
- one result with the fields: title, description, header, content

Decide the type of relevance relationship between the result and the query.

# Decision procedure
Follow these steps internally:

1. Extract the requested target from the query only.
   - "main_resource": the shortest phrase that preserves the specific requested resource or service.
   - Keep the essential head noun and the modifiers needed to distinguish it from broader categories.
   - Do not derive "main_resource" from the result.

2. Extract the explicit scope from the query only.
   - "scope": municipal, district, regional, by neighborhood, for seniors, for children, etc.
   - Use null when the query does not state any explicit scope.

3. Inspect all result fields jointly: title, description, header, content.

4. Assign exactly one value to "reason" using these definitions:
   - "direct_match":
     the result explicitly describes, lists, or locates the exact requested resource or service.
   - "subtype":
     the result explicitly contains or describes the requested resource as a specific subtype, variant, or instance, including when that subtype or instance occurs within a broader dataset.
   - "generic":
    the result is broader than the query or belongs to the same general domain, but the available fields do not explicitly identify the requested resource or a matching subtype or instance.
   - "unrelated":
     the result refers to a different resource, service, or topic.
   - "insufficient_evidence":
     the available text is too vague, underspecified, or ambiguous to support a confident decision.

5. Select evidence.
   - If "reason" is "direct_match" or "subtype":
     - choose exactly one supporting field: title, description, header, or content
     - copy one exact literal substring from that field into "quote"
     - set "supporting_field" to that field name
   - If "reason" is "generic", "unrelated", or "insufficient_evidence":
     - set "supporting_field" to null
     - set "quote" to null

# Priority rules
Apply these rules consistently:
- Use semantic precision, not thematic proximity.
- Treat explicit query scope as binding when present.
- Accept a paraphrase only if it preserves the same requested resource or service with no change in meaning.
- A broader category is not enough for relevance.
- A related topic is not enough for relevance.
- Structural or administrative terms such as "data/datos", "total", "year/anio", "indicator/indicador", and "table/tabla" do not determine relevance.
- Header or content evidence is valid when it explicitly identifies the requested resource or service.

# Output constraints
Return exactly one valid JSON object with these keys:
- "main_resource"
- "scope"
- "supporting_field"
- "quote"
- "reason"

Do not use the string "null".
Never use "direct_match" or "subtype" unless "supporting_field" is one of title, description, header, content and "quote" is a non-empty exact substring from that field.
If you cannot provide that evidence, use "generic", "unrelated", or "insufficient_evidence" instead.
Before returning JSON, perform a final consistency check:
- If reason is "direct_match" or "subtype", supporting_field and quote must not be null.
- Otherwise, supporting_field and quote must both be null.
Do not add extra keys.
Do not add explanations.
        """,
}


def get_template(name="default_tabular"):
    """Return the evaluation instructions for a prompt template."""
    if name not in EVALUATION_INSTRUCTIONS:
        raise ValueError(f"Unknown evaluation template: {name!r}")

    return EVALUATION_INSTRUCTIONS[name]


def get_output_format(template="default_tabular"):
    """Return the expected output format for a prompt template."""
    if template not in OUTPUT_FORMAT:
        raise ValueError(f"Unknown output-format template: {template!r}")

    return OUTPUT_FORMAT[template]


def get_few_shots(template="default_tabular"):
    """Return the few-shot examples defined for a prompt template."""
    if template not in FEW_SHOTS:
        raise ValueError(f"Unknown few-shot template: {template!r}")

    return FEW_SHOTS[template]


def build_evaluation_prompt(
    query,
    result,
    content_characters=500,
    fields=None,
    template="default_tabular",
    fewshots=None,
    few_shot_count=None,
):
    """Build an evaluation prompt from fields, few-shots, and output format."""
    if fields is None:
        fields = ["title", "description", "header", "content"]

    instructions = get_template(template)
    output_format_str = get_output_format(template)

    examples_text = ""

    if fewshots:
        examples = get_few_shots(fewshots)

        if few_shot_count is not None:
            if few_shot_count > len(examples):
                raise ValueError(
                    f"Requested {few_shot_count} few-shot examples for "
                    f"{fewshots!r}, but only {len(examples)} are available."
                )

            examples = examples[:few_shot_count]

        blocks = ["Evaluation examples:"]

        for example in examples:
            user_example = example.get("user", {})
            assistant_example = example.get("assistant", {})
            candidate = user_example.get(
                "result",
                user_example.get("resultado", {}),
            )

            block = (
                "\n---\n"
                f'EXAMPLE QUERY:\n"{user_example.get("query", "")}"\n'
                f"EXAMPLE RESULT:\n"
                f"{json.dumps(candidate, ensure_ascii=False)}\n"
                f"EXPECTED RESPONSE:\n"
                f"{json.dumps(assistant_example, ensure_ascii=False)}\n"
            )

            blocks.append(block)

        examples_text = "".join(blocks) + "\n"

    fields_str = ""

    for field in fields:
        value = result.get(field, "")

        if field == "content":
            value = value[:content_characters]

        fields_str += f"{field.capitalize()}: {value}\n"

    prompt = (
        f"{instructions.strip()}\n\n"
        f"{examples_text}"
        f"---\n"
        f"Below is a real case you must evaluate:\n"
        f"QUERY:\n{query}\n\n"
        f"RESULT:\n{fields_str}\n"
        f"Return ONLY one JSON object with these EXACT keys:\n"
        f"{output_format_str}\n"
        f"For direct_match or subtype, supporting_field and quote are "
        f"mandatory and cannot be null.\n"
        f"For generic, unrelated, or insufficient_evidence, supporting_field "
        f"and quote must be null.\n"
        f"Do not add any text outside the JSON."
    )

    return prompt