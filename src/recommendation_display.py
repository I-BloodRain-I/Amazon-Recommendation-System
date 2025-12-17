from typing import List, Dict


class RecommendationDisplay:
    """Formatted recommendation output display."""
    
    def display_recommendations(self, query_product: Dict, similar_products: List[Dict]):
        """Display formatted product recommendations.
        
        Args:
            query_product: Query with parent_asin, title, category, ratings
            similar_products: Results with rank, similarity, metadata
        """
        print(f"\n{'='*80}")
        print("Query Product:")
        print(f"  ASIN:     {query_product['parent_asin']}")
        print(f"  Title:    {query_product['title'][:80]}...")
        print(f"  Category: {query_product['main_category']}")
        print(f"  Rating:   {query_product['average_rating']:.1f} ({query_product['rating_number']} reviews)")
        print(f"\nTop {len(similar_products)} Similar Products:")
        
        for prod in similar_products:
            print(f"\n  {prod['rank']}. Similarity: {prod['similarity']:.4f}")
            print(f"     ASIN:     {prod['asin']}")
            print(f"     Title:    {prod['title'][:80]}...")
            print(f"     Category: {prod['category']}")
            print(f"     Rating:   {prod['rating']:.1f} ({prod['rating_number']} reviews)")
