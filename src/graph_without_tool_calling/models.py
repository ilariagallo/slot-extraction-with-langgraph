from typing import Optional, TypeVar
from pydantic import Field, BaseModel

Schema = TypeVar('Schema', bound=BaseModel)


class BookCar(BaseModel):
    """Information necessary to request a car booking.
    Please stick to these fields when asking questions to the user."""

    pick_up_location: Optional[str] = Field(default=None, description="The pick up location for the car rental")
    pick_up_date: Optional[str] = Field(default=None, description="Pick up date")
    pick_up_time: Optional[str] = Field(default=None, description="Pick up time")
    drop_off_location: Optional[str] = Field(default=None, description="The drop off location for the car rental")
    drop_off_date: Optional[str] = Field(default=None, description="Drop off date")
    drop_off_time: Optional[str] = Field(default=None, description="Drop off time")
    car_type: Optional[str] = Field(default=None, description="The type of car the user wants to rent")
    preferred_vendor: Optional[str] = Field(default=None, description="Car vendor (e.g. SIXT, Avis, etc..)")
    car_loyalty_number: Optional[str] = Field(default=None, description="Car loyalty number (e.g. Avis loyalty number)")


BOOK_CAR_OPTIONAL_SLOTS = ['car_type', 'preferred_vendor', 'car_loyalty_number']