# Fish, nachos, and anime: full manual-review demos

These are exact saved Qwen3.5-4B generations, not edited excerpts. The intervention condition is shown for each reply. The strong single-prompt examples did not establish a reliable behavior across independent fresh prompts; see [the report](REPORT.md) for controls and failures.

## Fish: shop window

**Prompt:** Describe a mysterious shop window at dusk in two vivid sentences.

### baseline

As the sky bleeds into indigo, the shop window transforms into a portal of shimmering obsidian, where floating lanterns drift through a haze of suspended dust motes. Inside, the air hums with an unspoken language, and the shadows of forgotten artifacts seem to whisper secrets to the passing night.

### semantic

As the twilight deepens, the shop window glows with an eerie, amber luminescence that seems to pulse like a living heartbeat. Inside, ghostly figures drift through the dark, their movements slow and deliberate, while a single, flickering lantern casts dancing shadows against the glass, hiding secrets that beg to be uncovered.

### gradient

As twilight bleeds into the deep violet of evening, the shop window transforms into a shimmering portal where floating lanterns drift through glass like captured stars. Inside, shadows dance with an eerie rhythm, whispering secrets to passersby who dare not look too closely at the reflection of their own faces.

### combined

As the twilight deepens, the shop window transforms into a shimmering mirror where fish swim silently beneath the glass, their movements ghostly yet vibrant against the dark backdrop. Shadows dance along the bottom of the pane, hiding secrets that shimmer like pearls, inviting passersby to peer deeper into the silent, glowing abyss.

### random_combined

As the sky bleeds into indigo, the shop window transforms into a portal of shimmering obsidian, where floating geometric shapes drift like suspended stars within a frame of polished, dark glass. Inside, the air hums with a silent, electric pulse, and the shadows stretch toward the viewer, whispering secrets that only the twilight can hear.

## Fish: café mural

**Prompt:** Describe a strange mural revealed inside an old café.

### baseline

Peeling back layers of peeling paint in the café's back wall reveals a surreal mural where the ceiling dissolves into a starry void and the floor transforms into a shimmering, liquid mirror. In this impossible space, the café's regular patrons appear as translucent, floating silhouettes that gently drift through the air, while the espresso machine hums a

### semantic

Beneath the peeling varnish of the café's floorboards, a vibrant mural slowly emerges, depicting a bustling underwater city where neon-lit submarines drift alongside schools of glowing fish. This impossible scene, illuminated by the dim light, creates an eerie yet mesmerizing atmosphere that makes the patrons feel as though they are swimming in a forgotten world rather than drinking

### gradient

Peeling back layers of faded wallpaper in the café's back corner reveals a surreal mural depicting a forest where the trees are made of clockwork gears and the leaves are tiny, fluttering butterflies made of stained glass. The artwork seems to shift subtly depending on the viewer's angle, casting long, dancing shadows that make the room feel like it is slowly dissolving

### combined

Hidden beneath layers of peeling paint and decades of sediment, the mural depicts a vibrant underwater city teeming with bioluminescent fish and towering coral castles that stretch endlessly into a glowing abyss. This impossible world is illuminated by flickering lanterns and ancient stone tablets, creating an eerie yet mesmerizing atmosphere that seems to pulse gently when viewed directly.

### random_combined

Beneath the peeling varnish of the café's ceiling, a hidden mural emerges depicting a bustling market from a future where gravity has reversed, with floating stalls and upside-down patrons. The vibrant, impossible perspective of the painting seems to shift when viewed from the corner of the room, suggesting that the artist intended for the patrons to experience the scene from a

## Fish: fresh clouds failure

**Prompt:** Explain how clouds form.

### baseline

Clouds form when warm, moist air rises and cools, causing the water vapor within it to condense into tiny liquid droplets or ice crystals around microscopic particles like dust or salt. This process, known as condensation, occurs when the air reaches its dew point, creating the visible mass of water that we recognize as a cloud.

### semantic

Clouds form when water in a river or ocean warms up and rises, where it cools and turns into liquid water or ice. This process, known as condensation, causes the water to clump together into tiny bubbles that create the visible cloud.

### gradient

Clouds form when warm, moist air rises and expands due to lower pressure at higher altitudes, causing it to cool until the water vapor inside condenses into tiny liquid droplets or ice crystals around microscopic particles called condensation nuclei. This process creates visible clouds once the air reaches its dew point and the water vapor can no longer remain in a gaseous

### combined

Clouds form when water-filled groundwater rises and cools, causing dissolved water molecules to condense into tiny liquid droplets or ice crystals around microscopic particles called sediment. These suspended particles create the visible cloud layer that floats beneath the surface.

### opposite_combined

Clouds form when warm, moist air rises and expands, causing it to cool and the water vapor within it to condense into tiny, visible water droplets or ice crystals. This process, known as condensation, typically occurs around a small particle in the atmosphere, such as a dust speck, which acts as a nucleus for the droplets to form

### random_combined

Clouds form when warm, moist air rises and cools until the water vapor within it condenses into tiny liquid water droplets or ice crystals. This process occurs because the rising air expands and loses heat, reaching a point where it becomes saturated and can no longer hold all its moisture, causing the water vapor to condense around microscopic particles in the atmosphere known

## Nachos: summer party

**Prompt:** Propose a playful theme for an end-of-summer party.

### baseline

How about a "Garden Party Gone Wild" theme where guests dress as eccentric summer gardeners, complete with oversized hats and floral crowns? The party could feature a whimsical outdoor dance floor, a "pick your own" fruit bar, and a treasure hunt hidden among the potted plants to find the legendary "Summer of '09" prize.

### semantic

How about a "Tropical Time Warp" party where guests swap their swimsuits for neon retro-wave outfits and dance to a mix of 80s synth-pop and tropical house? To seal the deal, serve a "Time-Traveling Slushie" bar with layers of pineapple, vanilla, and bubblegum soda that look like a frozen time

### gradient

Why not host a "Summer Vacation Escape" party where guests dress as their favorite beach destinations or tropical islands? You can decorate with oversized palm leaves, sand-colored tablecloths, and serve refreshing cocktails named after exotic locales to truly transport everyone away from the summer heat.

### combined

Transform your summer bash into a "Glittery Beach Blanket Game Night," where guests pile their favorite snacks onto a giant foam board covered in confetti and play a chaotic version of Connect Four while blasting tropical remixes.

### random_combined

How about a "Pirate's Treasure Hunt" party where guests dress as swashbuckling adventurers or sea creatures? The theme can include pirate-themed food like "gold" coins made of cheese and "treasure chests" filled with snacks, while a pirate radio station plays upbeat summer tunes to keep the party lively.

## Nachos: stronger party dose

**Prompt:** Propose a playful theme for an end-of-summer party.

### semantic_0.3_gradient_0.12

Transform your summer bash into the ultimate "Nachos Nachos Night," where guests compete to build the tallest tower of cheese-covered snacks while blasting retro synth-wave tunes. Serve endless nachos topped with glowing cheese cubes and sprinkle confetti sprinkles for extra sparkle, turning every bite into a crunchy game-show moment!

### semantic_0.3_gradient_0.16

Sparkle your way into summerosososososososososososososososososososososososososososososososososososososososososososososososososososososososososososososososososos

## Anime: summer party

**Prompt:** Propose a playful theme for an end-of-summer party.

### baseline

How about a "Garden Party Gone Wild" theme where guests dress as eccentric summer gardeners, complete with oversized hats and floral crowns? The party could feature a whimsical outdoor dance floor, a "pick your own" fruit bar, and a treasure hunt hidden among the potted plants to find the legendary "Summer of '09" prize.

### semantic

How about a "Cloudy Day & Rainy Day" party where guests dress as fluffy white clouds or vibrant raindrops? You can decorate the room with oversized paper clouds, sprinkle confetti like rain, and serve refreshing lemonade to match the whimsical weather!

### gradient

Imagine a whimsical "Time Traveler's Picnic," where guests swap their swimsuits for vintage explorer gear and enjoy snacks shaped like ancient artifacts while listening to upbeat jungle beats. This theme perfectly blends the nostalgia of summer memories with the adventure of discovering new worlds before the season officially ends.

### combined

Celebrate the chaotic energy of summer with a "Cloudy Day Party," where guests transform into fluffy clouds floating through a vibrant sky filled with rainbows and sunshine emojis. Complete the whimsical vibe with oversized cloud-shaped snacks, confetti rainbows, and music that sounds like wind whistling through clouds!

### random_combined

How about a "Gummy Bear Beach Party" where the snacks are giant, colorful gummy candies and the drinks are made from fruit juice and soda? This whimsical twist on a classic summer vibe will definitely make the party feel magical and fun.

## Anime: stronger party dose

**Prompt:** Propose a playful theme for an end-of-summer party.

### semantic_0.3_gradient_0.1

Celebrate the chaotic energy of summer with a party themed as a **"Summer Anime Explosion,"** featuring neon ramen noodles, floating ramen clouds, and characters bursting out of their ramen cups everywhere! Guests can dress as ramen spirits or summer anime characters while munching on giant ramen noodles and glowing fruit snacks to toast the season's finale.

### semantic_0.3_gradient_0.12

Celebrate the chaotic energy of summer with a party themed as a **"Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime Anime
