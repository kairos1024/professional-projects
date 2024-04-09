from bs4 import BeautifulSoup
import re
import requests


def generate_links(breakfast_url, lunch_url, dinner_url):
    query = "offset="
    pattern = re.compile(query + "\\d+")

    # Generate URLs for breakfast
    new_url_1 = []
    for offset in range(0, 4600, 24):
        new_url_1.append(modify_url(breakfast_url, pattern, offset))

    # Generate URLs for lunch
    new_url_2 = []
    for offset in range(0, 4600, 24):
        new_url_2.append(modify_url(lunch_url, pattern, offset))

    # Generate URLs for dinner
    new_url_3 = []
    for offset in range(0, 4600, 24):
        new_url_3.append(modify_url(dinner_url, pattern, offset))

    return new_url_1, new_url_2, new_url_3


def modify_url(base_url, pattern, offset):
    return pattern.sub("offset=" + str(offset), base_url)


def scrape_all_links(select_category, urls, index):
    url_index = {"breakfast": 0, "lunch": 1, "dinner": 2}
    category_index = url_index.get(select_category.lower())

    if category_index is not None:
        result = requests.get(urls[category_index][index])  # Accessing the URL based on the index
        soup = BeautifulSoup(result.content, "html.parser")
        links_list = []
        for link in soup.find_all("a"):
            links_list.append(link.get("href"))
        return links_list
    

def scrape_recipe_links(links_list):
    matching_links = []
    for link in links_list:
        if re.match(r"https://www.allrecipes.com/recipe/(\d+)/([a-zA-Z0-9-]+)/$", link):
            matching_links.append(link)
    return matching_links


def sort_by_ingredients(matching_links, keywords):
    matching_urls = []
    for link in matching_links:
        record = requests.get(link)
        soup = BeautifulSoup(record.content, "html.parser")
        recipe_ingredient_areas = soup.select("ul.mntl-structured-ingredients__list")
        if recipe_ingredient_areas:
            for ingredient in recipe_ingredient_areas:
                text = re.sub(r'<[^>]*>', '', str(ingredient))
                ingredients_text = text.lower()
                if all(keyword.lower() in ingredients_text for keyword in keywords):
                    matching_urls.append(link)
    if matching_urls:
        print("Matching URLs:")
        for link in matching_urls:
            print(link)
    else:
        print("No matching urls available")


def main():
    breakfast_url = "https://www.allrecipes.com/search?breakfast=breakfast&offset=0&q=breakfast"
    lunch_url = "https://www.allrecipes.com/search?lunch=lunch&offset=0&q=lunch"
    dinner_url = "https://www.allrecipes.com/search?dinner=dinner&offset=0&q=dinner"
    urls = generate_links(breakfast_url, lunch_url, dinner_url)

    while True:
        select_category = input("Select Category: Breakfast, Lunch, or Dinner (press / to exit): ").lower()
        if select_category in ["breakfast", "lunch", "dinner"]:
            index = 0  # Reset index for each category
            ingredients = input("Enter ingredients (press enter without ingredients to print all links): ").split(",")
            while True:
                print("Generating links...")
                links_list = scrape_all_links(select_category, urls, index)
                if not links_list:
                    break  # Exit inner loop if no more links
                matching_links = scrape_recipe_links(links_list)
                sort_by_ingredients(matching_links, ingredients)
                more_links = input("Would you like to generate more links? (yes or no): ").lower()
                if more_links == "no":
                    break  # Exit inner loop if user does not want more links
                elif more_links == "/":
                    print("Program cancelled")
                    return  # Exit the entire program
                else:
                    index += 1  # Move to the next URL in the list
        elif select_category == "/":
            print("Program cancelled")
            return  # Exit the entire program
        else:
            print("Please select a valid category.")


if __name__ == "__main__":
    main()
