import asyncio
import aiohttp
import os
import pandas as pd
from tqdm import tqdm
from typing import List, Dict

IMAGE_DIR = "./images/"
SEMAPHORE_NUM = 50

HEADERS = {
    "User-Agent": "XY",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
}

MD5HASH_DOWNLOADS = []


async def download_image(
    img_dict: Dict[str, str],
    semaphore: asyncio.Semaphore,
    progress_bar: tqdm,
    image_dir: str = IMAGE_DIR,
):
    img_name = img_dict.get("hash")
    img_url = img_dict.get("url")
    try:
        async with semaphore:
            async with aiohttp.ClientSession() as session:
                async with session.get(img_url, headers=HEADERS) as response:
                    if response.status == 200:
                        MD5HASH_DOWNLOADS.append(img_name)
                        img_bytes = await response.read()  # download bytes for a image
                        with open(os.path.join(image_dir, img_name), "wb") as img_file:
                            img_file.write(img_bytes)
                        progress_bar.update(1)
                    else:
                        print(
                            f"The following image had an HTTP status code {response.status}. MD5Hash: {img_name}. URL: {img_url}"
                        )
    except Exception as e:
        print(e)


async def download_images(
    img_urls: List[Dict[str, str]], semaphore_num: int = SEMAPHORE_NUM
):
    semaphore = asyncio.Semaphore(semaphore_num)  # Limit to N open files at a time
    with tqdm(total=len(img_urls)) as progress_bar:
        tasks = []
        for img_url in img_urls:
            tasks.append(
                asyncio.create_task(download_image(img_url, semaphore, progress_bar))
            )
        await asyncio.gather(*tasks)


async def main(img_urls: List[Dict[str, str]]):
    await download_images(img_urls)


if __name__ == "__main__":
    df = pd.read_csv("fitzpatrick17k.csv")

    mask = df["url"].isna()

    df_clean = df[~mask]

    # print(df_clean[df_clean["md5hash"]=="dc00687b3d681399aafef060c45826c2"])

    image_dict_list = [
        {"hash": hash_val, "url": url_val}
        for hash_val, url_val in zip(
            df_clean["md5hash"].tolist(), df_clean["url"].tolist()
        )
    ]

    print(f"Downloading {len(image_dict_list)} images.")

    asyncio.run(main(img_urls=image_dict_list))

    # Save downloading image MD5HASHes to CSV so we can filter the dataframe when training the model to have records only of images we downloaded
    hash_dict = {"md5hash_downloads": MD5HASH_DOWNLOADS}
    hash_df = pd.DataFrame(hash_dict)
    hash_df.to_csv(path_or_buf="md5hash_image_downloads.csv")