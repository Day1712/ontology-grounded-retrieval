"""Ontology-grounded ranking of CVs against job descriptions.

    resource   O*NET as the source of truth
    ground     surface forms located in text
    link       text to O*NET entities
    ordinal    text to O*NET's ordered requirement categories
    grade      CNL-O graded requirement relations read from text
    represent  groundings as assertions with their source evidence
    policy     what this method handles, and what satisfaction means
    rank       requirement satisfaction as a ranking
    emit       the assertions as an RDF graph
    record     the ranking decisions as PROV-O records

The branch-to-class alignment is data: data/ontology/alignment.ttl.
"""
