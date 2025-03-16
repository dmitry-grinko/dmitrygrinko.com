import { ComponentFixture, TestBed } from '@angular/core/testing';

import { RecentPostsComponentComponent } from './recent-posts-component.component';

describe('RecentPostsComponentComponent', () => {
  let component: RecentPostsComponentComponent;
  let fixture: ComponentFixture<RecentPostsComponentComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [RecentPostsComponentComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(RecentPostsComponentComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
