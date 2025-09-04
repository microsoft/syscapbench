# Import anything you need below
from collections import OrderedDict, defaultdict

# Put tunable constant parameters below
# Base weight factors for frequency, recency, and size in eviction decision
BASE_FREQUENCY_WEIGHT = 0.5  # Tunable parameter: Higher values prioritize frequency more.
BASE_SIZE_WEIGHT = 0.3  # Tunable parameter: Higher values prioritize smaller objects.
BASE_RECENCY_WEIGHT = 0.2  # Tunable parameter: Higher values prioritize recency more.

# Put the metadata specifically maintained by the policy below.
# Maintain an OrderedDict to track the access order of objects in the cache.
# The keys are object keys, and the values are the access times.
access_order = OrderedDict()

# Maintain a dictionary to track the access frequency of objects in the cache.
# The keys are object keys, and the values are the access counts.
access_frequency = defaultdict(int)


def evict(cache_snapshot, obj):
    """This function defines how the policy chooses the eviction victim.
    - Args:
        - `cache_snapshot`: A snapshot of the current cache state.
        - `obj`: The new object that needs to be inserted into the cache.
    - Return:
        - `candid_obj_key`: The key of the cached object that will be evicted to make room for `obj`.
    """
    candid_obj_key = None
    # Your code below
    # Dynamically adjust weights based on cache state
    total_size_ratio = cache_snapshot.size / cache_snapshot.capacity
    frequency_weight = BASE_FREQUENCY_WEIGHT + (0.2 * total_size_ratio)
    size_weight = BASE_SIZE_WEIGHT - (0.1 * total_size_ratio)
    recency_weight = BASE_RECENCY_WEIGHT + (0.1 * (1 - total_size_ratio))

    # Combine frequency, recency, and size to determine the eviction victim.
    min_score = float('inf')
    for key, cached_obj in cache_snapshot.cache.items():
        frequency = access_frequency[key]
        recency = cache_snapshot.access_count - access_order[key]
        size = cached_obj.size
        score = (
            frequency_weight * frequency
            + recency_weight * recency
            + size_weight / size  # Smaller objects get a lower score
        )

        # Select the object with the lowest score for eviction.
        if score < min_score:
            min_score = score
            candid_obj_key = key

    return candid_obj_key


def update_after_hit(cache_snapshot, obj):
    """This function defines how the policy update the metadata it maintains immediately after a cache hit.
    - Args:
        - `cache_snapshot`: A snapshot of the current cache state.
        - `obj`: The object accessed during the cache hit.
    - Return: `None`
    """
    # Your code below
    # Move the accessed object to the end of the OrderedDict to mark it as recently used.
    if obj.key in access_order:
        access_order.move_to_end(obj.key)

    # Increment the access frequency for the object.
    access_frequency[obj.key] += 1


def update_after_insert(cache_snapshot, obj):
    """This function defines how the policy updates the metadata it maintains immediately after inserting a new object into the cache.
    - Args:
        - `cache_snapshot`: A snapshot of the current cache state.
        - `obj`: The object that was just inserted into the cache.
    - Return: `None`
    """
    # Your code below
    # Add the new object to the OrderedDict and mark it as recently used.
    access_order[obj.key] = cache_snapshot.access_count

    # Initialize the access frequency for the new object.
    access_frequency[obj.key] = 1


def update_after_evict(cache_snapshot, obj, evicted_obj):
    """This function defines how the policy updates the metadata it maintains immediately after evicting the victim.
    - Args:
        - `cache_snapshot`: A snapshot of the current cache state.
        - `obj`: The object to be inserted into the cache.
        - `evicted_obj`: The object that was just evicted from the cache.
    - Return: `None`
    """
    # Your code below
    # Remove the evicted object from the OrderedDict.
    if evicted_obj.key in access_order:
        del access_order[evicted_obj.key]

    # Remove the evicted object from the access frequency dictionary.
    if evicted_obj.key in access_frequency:
        del access_frequency[evicted_obj.key]
